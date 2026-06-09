import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from utils.hf_artifacts import download_dataset_files


FEATURE_COLUMNS = [
    "x1",
    "y1",
    "z1",
    "R",
    "Bx_U",
    "By_U",
    "Bz_U",
    "M",
    "RhoO",
    "H0",
    "n0",
]
TARGET_COLUMNS = ["delta_Bx", "delta_By", "delta_Bz"]
META_COLUMNS = ["sample", "flyby", "traj_id", "dt"]


class PaddedTrajectoryDataset(Dataset):
    """Trajectory dataset that stores per-trajectory tensors and valid masks.

    Args:
        df: Dataframe containing at least `FEATURE_COLUMNS`, `TARGET_COLUMNS`,
            and `traj_id`.
    """

    def __init__(self, df, feature_columns=None, target_columns=None):
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.target_columns = target_columns or TARGET_COLUMNS
        self.trajectories = []
        for _, group in df.groupby("traj_id", sort=False):
            X = torch.from_numpy(group[self.feature_columns].to_numpy(dtype=np.float32, copy=False))
            y = torch.from_numpy(group[self.target_columns].to_numpy(dtype=np.float32, copy=False))
            mask = torch.ones(len(X), dtype=torch.bool)
            self.trajectories.append((X, y, mask))

    def __len__(self):
        """Return number of trajectories."""
        return len(self.trajectories)

    def __getitem__(self, idx):
        """Return one `(features, targets, mask)` trajectory tuple."""
        return self.trajectories[idx]


def pad_collate(batch):
    """Pad variable-length trajectories into batch-first tensors.

    Args:
        batch: Iterable of `(X, y, mask)` tuples from `PaddedTrajectoryDataset`.

    Returns:
        Tuple `(Xs, ys, masks)` where each tensor is padded to max sequence length.
    """
    Xs, ys, masks = zip(*batch)
    Xs = pad_sequence(Xs, batch_first=True)
    ys = pad_sequence(ys, batch_first=True)
    masks = pad_sequence(masks, batch_first=True)
    return Xs, ys, masks


def _sort_by_trajectory(df):
    """Sort dataframe by trajectory/time for deterministic sequence construction."""
    return df.sort_values(["traj_id", "dt"]).reset_index(drop=True)


def _scale_splits(train_df, val_df, test_df):
    """Fit scaler on train split and transform train/val/test consistently.

    Args:
        train_df: Unscaled training dataframe.
        val_df: Unscaled validation dataframe.
        test_df: Unscaled test dataframe.

    Returns:
        Tuple `(train_scaled, val_scaled, test_scaled, scaler)`.
    """
    scaler = StandardScaler().fit(train_df[FEATURE_COLUMNS])

    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    train_scaled[FEATURE_COLUMNS] = scaler.transform(train_df[FEATURE_COLUMNS])
    val_scaled[FEATURE_COLUMNS] = scaler.transform(val_df[FEATURE_COLUMNS])
    test_scaled[FEATURE_COLUMNS] = scaler.transform(test_df[FEATURE_COLUMNS])

    return train_scaled, val_scaled, test_scaled, scaler


def _build_loaders(train_df, val_df, test_df, train_batch_size=8, eval_batch_size=1):
    """Create dataloaders from split dataframes.

    Args:
        train_df: Scaled training dataframe.
        val_df: Scaled validation dataframe.
        test_df: Scaled test dataframe.
        train_batch_size: Batch size for training loader.
        eval_batch_size: Batch size for validation/test loaders.

    Returns:
        Tuple `(train_loader, val_loader, test_loader)`.
    """
    train_ds = PaddedTrajectoryDataset(train_df)
    val_ds = PaddedTrajectoryDataset(val_df)
    test_ds = PaddedTrajectoryDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, collate_fn=pad_collate)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, collate_fn=pad_collate)
    return train_loader, val_loader, test_loader


def prepare_data_bundle(
    hf_dataset_repo="reddysachin/LEAP_dataset",
    output_dir="data/out",
    train_batch_size=8,
    eval_batch_size=1,
):
    """Build train/val/test data artifacts from published HF splits.

    Args:
        hf_dataset_repo: Hugging Face dataset repo id.
        output_dir: Directory for persisted scaler/test artifacts.
        train_batch_size: Batch size for training loader.
        eval_batch_size: Batch size for validation/test loaders.

    Returns:
        Dict containing loaders, scaled test dataframe, fitted scaler, and paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[Data] Fetching dataset artifacts from Hugging Face...")
    paths = download_dataset_files(repo_id=hf_dataset_repo)
    print("[Data] Reading parquet files...")
    train_df = _sort_by_trajectory(pd.read_parquet(paths["train"]))
    val_df = _sort_by_trajectory(pd.read_parquet(paths["validation"]))
    test_df = _sort_by_trajectory(pd.read_parquet(paths["test"]))
    print(f"[Data] Rows -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    print("[Data] Scaling features using training split...")
    train_scaled, val_scaled, test_scaled, scaler = _scale_splits(train_df, val_df, test_df)
    print("[Data] Building DataLoaders...")
    train_loader, val_loader, test_loader = _build_loaders(
        train_scaled,
        val_scaled,
        test_scaled,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
    )
    print(
        f"[Data] Trajectories -> train: {len(train_loader.dataset)}, "
        f"val: {len(val_loader.dataset)}, test: {len(test_loader.dataset)}"
    )

    scaler_path = output_dir / "scaler.pkl"
    test_data_path = output_dir / "test_data.parquet"

    with open(scaler_path, "wb") as file:
        pickle.dump(scaler, file)
    test_scaled.to_parquet(test_data_path, index=False)
    print(f"[Data] Saved scaler: {scaler_path}")
    print(f"[Data] Saved scaled test split: {test_data_path}")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "df_test_scaled": test_scaled,
        "scaler": scaler,
        "scaler_path": str(scaler_path),
        "test_data_path": str(test_data_path),
        "source_paths": paths,
    }
