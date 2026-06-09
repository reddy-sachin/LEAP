from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from data.pipeline import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    PaddedTrajectoryDataset,
    pad_collate,
    prepare_data_bundle,
)
from evaluate.evaluate import compute_component_errors, save_component_errors, compute_crude_mae
from loss.loss import ScienceLoss, monitoring_loss
from model.transformer import Lightweight
from utils.config import load_config
from utils.hf_artifacts import download_dataset_test_file, download_model_artifacts

try:
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:  # pragma: no cover - sklearn may not be installed in some environments
    InconsistentVersionWarning = None

# Keep CLI output clean for reproducibility runs.
if InconsistentVersionWarning is not None:
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")
_FEATURE_INDEX = {name: idx for idx, name in enumerate(FEATURE_COLUMNS)}


def _assert_required_loss_features(feature_columns, context="runtime"):
    """Fail fast if required loss-driving features are missing."""
    required_any = [["R", "R1"]]
    required_all = ["x1", "y1", "z1", "Bx_U", "By_U", "Bz_U"]

    missing_all = [c for c in required_all if c not in feature_columns]
    if missing_all:
        raise ValueError(f"[{context}] Missing required features: {missing_all}")

    if not any(any_name in feature_columns for any_name in required_any[0]):
        raise ValueError(f"[{context}] Missing required radial feature: one of {required_any[0]}")


def maybe_generate_plot(results_csv_path, explicit_plot_path="", skip_plot=False):
    """Optionally generate Figure 3 from a saved results CSV.

    Args:
        results_csv_path: Path to evaluation CSV output.
        explicit_plot_path: Optional override for figure output path.
        skip_plot: If True, do not generate plots.
    """
    if skip_plot:
        print("[Plot] Skipped (requested by --skip-plot).")
        return

    save_path = explicit_plot_path.strip()
    if not save_path:
        save_path = "assets/figure3.png"

    try:
        from evaluate.plots import FigureThreePlotter

        print(f"[Plot] Generating figure from: {results_csv_path}")
        test_set = pd.read_csv(results_csv_path)
        fig3 = FigureThreePlotter()
        radial_bounds = fig3.set_radial_bounds(test_set)
        fig3.plot_combined_hist_radial(test_set, radial_bounds, save_path)
        print(f"[Plot] Saved figure: {save_path}")
    except Exception as exc:
        print(f"[Plot] Skipped due to plotting error: {exc}")


def resolve_runtime_devices(device_pref):
    """Resolve train/validation/evaluation devices.

    Args:
        device_pref: Device preference (`auto|cuda|mps|cpu`).

    Returns:
        Tuple `(train_device, val_device, eval_device)`.
    """
    pref = device_pref.lower()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()

    if pref == "mps":
        train_device = torch.device("mps") if mps_available else torch.device("cpu")
    elif pref == "cuda":
        train_device = torch.device("cuda") if cuda_available else torch.device("cpu")
    elif pref == "cpu":
        train_device = torch.device("cpu")
    else:
        # Auto policy requested:
        # 1) MPS train + CPU val/eval
        # 2) CUDA train/val + CPU eval
        # 3) CPU for all
        if mps_available:
            train_device = torch.device("mps")
        elif cuda_available:
            train_device = torch.device("cuda")
        else:
            train_device = torch.device("cpu")

    if train_device.type == "mps":
        val_device = torch.device("cpu")
    else:
        val_device = train_device
    eval_device = torch.device("cpu")
    return train_device, val_device, eval_device


def _feature_idx(name):
    """Return a feature index from `FEATURE_COLUMNS`.

    Args:
        name: Feature column name.

    Returns:
        Integer index of that feature in `FEATURE_COLUMNS`.
    """
    if name not in _FEATURE_INDEX:
        raise ValueError(f"Missing feature '{name}' in FEATURE_COLUMNS: {FEATURE_COLUMNS}")
    return _FEATURE_INDEX[name]


def _loss_feature_indices():
    """Return feature indices needed by science loss.

    Returns:
        Tuple `(coord_idx, r_idx, bj_idx)` where:
        - `coord_idx` indexes x/y/z coordinates,
        - `r_idx` indexes radius feature,
        - `bj_idx` indexes background field features.
    """
    coord_idx = [_feature_idx("x1"), _feature_idx("y1"), _feature_idx("z1")]
    r_idx = _feature_idx("R") if "R" in FEATURE_COLUMNS else _feature_idx("R1")
    bj_idx = [_feature_idx("Bx_U"), _feature_idx("By_U"), _feature_idx("Bz_U")]
    return coord_idx, r_idx, bj_idx


def _batch_science_loss(pred, target, inputs, mask, criterion, coord_idx, r_idx, bj_idx):
    """Compute science loss across all valid trajectories in a batch.

    Args:
        pred: Predicted field tensor `(B, T, 3)`.
        target: Target field tensor `(B, T, 3)`.
        inputs: Input feature tensor `(B, T, F)`.
        mask: Valid timestep mask `(B, T)`.
        criterion: Loss module with signature `(pred_i, target_i, BJ_i, R_i, coords_i)`.
        coord_idx: Coordinate feature indices.
        r_idx: Radius feature index.
        bj_idx: Background field feature indices.

    Returns:
        Scalar batch loss tensor.
    """
    losses = []
    for pred_i, target_i, feat_i, valid in zip(pred, target, inputs, mask):
        if valid.sum() < 2:
            continue

        pred_i = pred_i[valid]
        target_i = target_i[valid]
        feat_i = feat_i[valid]

        coords_i = feat_i[:, coord_idx]
        r_i = feat_i[:, r_idx]
        bj_i = feat_i[:, bj_idx]
        losses.append(criterion(pred_i, target_i, bj_i, r_i, coords_i))

    if not losses:
        return torch.zeros((), device=pred.device, requires_grad=True)
    return torch.stack(losses).mean()


def validate_on_cpu(model, val_loader, criterion, device, coord_idx, r_idx, bj_idx):
    """Run validation on CPU, then restore model device.

    Args:
        model: Torch model.
        val_loader: Validation dataloader.
        criterion: Loss module.
        device: Original training device to restore after validation.
        coord_idx: Coordinate feature indices.
        r_idx: Radius feature index.
        bj_idx: Background field feature indices.

    Returns:
        Mean validation loss.
    """
    model.cpu()
    model.eval()
    val_epoch_loss = 0.0

    with torch.no_grad():
        for val_X, val_y, val_mask in val_loader:
            src_key_padding_mask = ~val_mask
            val_pred = model(val_X, src_key_padding_mask=src_key_padding_mask)
            val_loss = _batch_science_loss(
                val_pred,
                val_y,
                val_X,
                val_mask,
                criterion,
                coord_idx,
                r_idx,
                bj_idx,
            )
            val_epoch_loss += val_loss.item()

    model.to(device)
    return val_epoch_loss / len(val_loader)


def validate_on_device(model, val_loader, criterion, device, coord_idx, r_idx, bj_idx):
    """Run validation on the same device as training.

    Args:
        model: Torch model.
        val_loader: Validation dataloader.
        criterion: Loss module.
        device: Runtime device.
        coord_idx: Coordinate feature indices.
        r_idx: Radius feature index.
        bj_idx: Background field feature indices.

    Returns:
        Mean validation loss.
    """
    model.eval()
    val_epoch_loss = 0.0

    with torch.no_grad():
        for val_X, val_y, val_mask in val_loader:
            val_X = val_X.to(device)
            val_y = val_y.to(device)
            val_mask = val_mask.to(device)

            src_key_padding_mask = ~val_mask
            val_pred = model(val_X, src_key_padding_mask=src_key_padding_mask)
            val_loss = _batch_science_loss(
                val_pred,
                val_y,
                val_X,
                val_mask,
                criterion,
                coord_idx,
                r_idx,
                bj_idx,
            )
            val_epoch_loss += val_loss.item()

    return val_epoch_loss / len(val_loader)


def train_model(config, train_loader, val_loader, model_dir, train_device, val_device):
    """Train model with science loss and early stopping.

    Args:
        config: Parsed runtime config.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        model_dir: Directory to save model checkpoints/history.
        train_device: Device for training.
        val_device: Device for validation.

    Returns:
        Tuple `(best_model, history_df)`.
    """
    device = train_device
    print(f"[Train] Train device: {train_device}")
    print(f"[Train] Validation device: {val_device}")
    print(f"[Train] Epochs: {config.n_epochs}, batch_size: {config.train_batch_size}, noise: {config.noise_level}")
    model = Lightweight(input_dim=len(FEATURE_COLUMNS), dropout=config.dropout).to(device)
    criterion = ScienceLoss(w_par=config.w_par, w_perp=config.w_perp, w_div=config.w_div)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    coord_idx, r_idx, bj_idx = _loss_feature_indices()

    train_losses = []
    bx_losses = []
    by_losses = []
    bz_losses = []
    val_losses = []
    learning_rates = []

    best_val_loss = float("inf")
    counter = 0

    best_weights_path = model_dir / config.weights_name

    use_cpu_for_val = (val_device.type == "cpu" and device.type != "cpu")
    saw_first_batch = False
    pos_idx = [_feature_idx("x1"), _feature_idx("y1"), _feature_idx("z1"), r_idx]

    for epoch in range(config.n_epochs):
        model.train()
        print(f"[Train] Epoch {epoch+1}/{config.n_epochs} started")
        epoch_loss = 0.0
        bx_loss_total, by_loss_total, bz_loss_total = 0.0, 0.0, 0.0

        total_batches = len(train_loader)
        for batch_idx, (batch_X, batch_y, batch_mask) in enumerate(train_loader, start=1):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)

            if config.noise_level > 0:
                # Add positional noise to x, y, z, R features.
                noised = batch_X.clone()
                pos_noise = torch.randn_like(noised[:, :, pos_idx]) * config.noise_level
                noised[:, :, pos_idx] += pos_noise
                batch_X = noised

            optimizer.zero_grad()
            pred = model(batch_X, src_key_padding_mask=~batch_mask)

            loss = _batch_science_loss(pred, batch_y, batch_X, batch_mask, criterion, coord_idx, r_idx, bj_idx)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            #for debugging: print shapes and valid point count for the first batch
            """if not saw_first_batch:
                valid_points = int(batch_mask.sum().item())
                print(
                    f"[Train] First batch shapes X={tuple(batch_X.shape)}, y={tuple(batch_y.shape)}, "
                    f"valid_points={valid_points}"
                )
                saw_first_batch = True"""

            if batch_idx % 50 == 0 or batch_idx == total_batches:
                print(
                    f"[Train] Epoch {epoch+1} progress: batch {batch_idx}/{total_batches} "
                    f"(running loss: {epoch_loss / batch_idx:.4f})"
                )

            pred_valid = pred[batch_mask]
            target_valid = batch_y[batch_mask]
            bx_loss_total += monitoring_loss(pred_valid[:, 0], target_valid[:, 0]).item()
            by_loss_total += monitoring_loss(pred_valid[:, 1], target_valid[:, 1]).item()
            bz_loss_total += monitoring_loss(pred_valid[:, 2], target_valid[:, 2]).item()

        scheduler.step()
        scheduler_lr = scheduler.get_last_lr()[0]
        learning_rates.append(scheduler_lr)

        train_loss = epoch_loss / len(train_loader)
        bx_loss = bx_loss_total / len(train_loader)
        by_loss = by_loss_total / len(train_loader)
        bz_loss = bz_loss_total / len(train_loader)

        train_losses.append(train_loss)
        bx_losses.append(bx_loss)
        by_losses.append(by_loss)
        bz_losses.append(bz_loss)

        if use_cpu_for_val:
            val_loss_avg = validate_on_cpu(model, val_loader, criterion, device, coord_idx, r_idx, bj_idx)
        else:
            val_loss_avg = validate_on_device(model, val_loader, criterion, device, coord_idx, r_idx, bj_idx)
        val_losses.append(val_loss_avg)

        print(
            f"[Train] Epoch {epoch+1} | Train Loss: {train_loss:.4f} | "
            f"Bx: {bx_loss:.4f} | By: {by_loss:.4f} | Bz: {bz_loss:.4f} | "
            f"Val Loss: {val_loss_avg:.4f} | LR: {scheduler_lr:.2e}"
        )

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            counter = 0
            torch.save(model.state_dict(), best_weights_path)
        else:
            counter += 1
            if counter >= config.patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {counter} epochs).")
                break

    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    final_weights_path = model_dir / config.final_weights_name
    torch.save(model.state_dict(), final_weights_path)

    history_df = pd.DataFrame(
        {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "bx_loss": bx_losses,
            "by_loss": by_losses,
            "bz_loss": bz_losses,
            "learning_rate": learning_rates,
        }
    )

    return model, history_df


def evaluate_model(model, loader, df_test_scaled, scaler, device, feature_columns):
    """Run model inference on test loader and build enriched results dataframe.

    Args:
        model: Trained model.
        loader: Test dataloader.
        df_test_scaled: Scaled dataframe used to build `loader`.
        scaler: Fitted scaler used for inverse-transform.
        device: Inference device.

    Returns:
        DataFrame containing predictions, targets, residuals and passthrough features.
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y, mask in loader:
            X = X.to(device)
            y = y.to(device)
            mask = mask.to(device)
            output = model(X, src_key_padding_mask=~mask)
            all_preds.append(output[mask].cpu().numpy())
            all_targets.append(y[mask].cpu().numpy())

    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    component_labels = [c.replace("delta_", "") for c in TARGET_COLUMNS]
    df = pd.DataFrame()
    for i, label in enumerate(component_labels):
        df[f"{label}_pred"] = predictions[:, i]
        df[f"{label}_true"] = targets[:, i]
        df[f"{label}_diff"] = df[f"{label}_pred"] - df[f"{label}_true"]

    df["B_true"] = np.sqrt(df["Bx_true"] ** 2 + df["By_true"] ** 2 + df["Bz_true"] ** 2)
    df["B_pred"] = np.sqrt(df["Bx_pred"] ** 2 + df["By_pred"] ** 2 + df["Bz_pred"] ** 2)
    df["B_diff"] = df["B_pred"] - df["B_true"]

    passthrough_cols = feature_columns + ["sample", "flyby", "traj_id"]
    for col in passthrough_cols:
        if col in df_test_scaled.columns:
            df[col] = df_test_scaled[col].values

    scaler_cols = list(getattr(scaler, "feature_names_in_", feature_columns))
    missing_for_inverse = [c for c in scaler_cols if c not in df.columns]
    if missing_for_inverse:
        raise ValueError(
            f"Cannot inverse-transform; missing columns from results dataframe: {missing_for_inverse}"
        )
    df[scaler_cols] = scaler.inverse_transform(df[scaler_cols])

    if "dt" in df_test_scaled.columns:
        df["dt"] = df_test_scaled["dt"].values
        dt_col = df.pop("dt")
        df.insert(0, "dt", dt_col)

    return df


def _load_eval_artifacts(config):
    """Load test split, scaler, and weights for `--eval-only`.

    Args:
        config: Parsed runtime config.

    Returns:
        Tuple `(test_loader, test_scaled_df, scaler, weights_path)`.
    """
    print("[Eval] Fetching test split + pretrained artifacts from Hugging Face...")
    test_path = download_dataset_test_file(repo_id=config.hf_dataset_repo)
    model_paths = download_model_artifacts(repo_id=config.hf_model_repo)
    test_df = pd.read_parquet(test_path)
    scaler_path = model_paths["scaler"]
    weights_path = model_paths["weights"]

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    test_scaled = test_df.copy()
    scaler_cols = list(getattr(scaler, "feature_names_in_", FEATURE_COLUMNS))
    missing_for_scaling = [c for c in scaler_cols if c not in test_scaled.columns]
    if missing_for_scaling:
        raise ValueError(
            f"Test split is missing scaler columns: {missing_for_scaling}. "
            f"Scaler expects: {scaler_cols}"
        )
    test_scaled[scaler_cols] = scaler.transform(test_scaled[scaler_cols])

    test_ds = PaddedTrajectoryDataset(test_scaled, feature_columns=scaler_cols, target_columns=TARGET_COLUMNS)
    test_loader = DataLoader(test_ds, batch_size=config.eval_batch_size, shuffle=False, collate_fn=pad_collate)

    return test_loader, test_scaled, scaler, weights_path, scaler_cols


def main():
    """CLI entry-point for training/evaluation workflows."""
    config = load_config()
    torch.manual_seed(config.seed)
    train_device, val_device, eval_device = resolve_runtime_devices(config.device)
    if config.plot_only:
        mode_label = "plot-only"
    elif config.eval_only:
        mode_label = "eval-only"
    else:
        mode_label = "train-eval"
    print(f"[Main] Mode: {mode_label}")
    print(f"[Main] Device policy -> train: {train_device}, val: {val_device}, eval: {eval_device}")
    print(f"[Main] Output dir: {config.output_dir}")
    _assert_required_loss_features(FEATURE_COLUMNS, context="train-eval schema")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / config.history_name
    results_path = output_dir / config.results_name
    metrics_path = output_dir / "metrics.txt"

    if config.plot_only:
        print("[Plot] Plot-only mode: generating figure from existing results CSV...")
        maybe_generate_plot(
            results_csv_path=str(results_path),
            explicit_plot_path=config.plot_path,
            skip_plot=False,
        )
        return

    if config.eval_only:
        test_loader, test_scaled, scaler, weights_path, eval_feature_columns = _load_eval_artifacts(config)
        _assert_required_loss_features(eval_feature_columns, context="eval-only scaler schema")
        model = Lightweight(input_dim=len(eval_feature_columns)).to(eval_device)
        print("[Eval] Loading pretrained weights on eval device...")
        state = torch.load(weights_path, map_location=eval_device)
        model.load_state_dict(state)

        print("[Eval] Calculating errors on test set...")
        test_results = evaluate_model(
            model,
            test_loader,
            test_scaled,
            scaler,
            eval_device,
            eval_feature_columns,
        )
        test_results.to_csv(results_path, index=False)
        compute_component_errors(test_results)
        save_component_errors(test_results, str(metrics_path))
        crude_mae = compute_crude_mae(test_results)
        print(f"Crude vector-component MAE (Bx/By/Bz mean): {crude_mae:.4f}")
        print(f"Saved test predictions: {results_path}")
        print(f"Saved evaluation metrics: {metrics_path}")
        maybe_generate_plot(
            results_csv_path=str(results_path),
            explicit_plot_path=config.plot_path,
            skip_plot=config.skip_plot,
        )

        return

    data_bundle = prepare_data_bundle(
        hf_dataset_repo=config.hf_dataset_repo,
        output_dir=config.output_dir,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
    )

    model, history = train_model(
        config,
        data_bundle["train_loader"],
        data_bundle["val_loader"],
        output_dir,
        train_device,
        val_device,
    )
    history.to_csv(history_path, index=False)
    print(f"Saved training history: {history_path}")

    if config.train_eval:
        test_results = evaluate_model(
            model,
            data_bundle["test_loader"],
            data_bundle["df_test_scaled"],
            data_bundle["scaler"],
            eval_device,
            FEATURE_COLUMNS,
        )
        test_results.to_csv(results_path, index=False)
        compute_component_errors(test_results)
        save_component_errors(test_results, str(metrics_path))
        print(f"Saved test predictions: {results_path}")
        print(f"Saved evaluation metrics: {metrics_path}")
        
        maybe_generate_plot(
            results_csv_path=str(results_path),
            explicit_plot_path=config.plot_path,
            skip_plot=config.skip_plot,
        )


if __name__ == "__main__":
    main()
