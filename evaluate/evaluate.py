# eval/evaluate.py
import numpy as np
import pandas as pd

def _component_mae_rmse(diff: pd.Series):
    """Compute MAE and RMSE from a 1D residual series."""
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return mae, rmse


def _component_mae(diff: pd.Series):
    """Compute MAE from a 1D residual series."""
    return float(np.mean(np.abs(diff)))


def compute_component_errors(df: pd.DataFrame):
    """
    Compute and print MAE for Bx, By, Bz and |B|.

    Args:
        df: Evaluation dataframe containing `*_diff` columns.
    """
    for comp in ["Bx", "By", "Bz", "B"]:
        diff = df[f"{comp}_diff"]
        mae = _component_mae(diff)
        print(f"{comp} MAE: {mae:.4f}")
    print("-" * 50)

def save_component_errors(df: pd.DataFrame, outfile: str):
    """
    Save MAE for Bx, By, Bz and |B| to a text file.

    Args:
        df: Evaluation dataframe containing `*_diff` columns.
        outfile: Output text filepath.
    """
    with open(outfile, "w") as f:
        for comp in ["Bx", "By", "Bz", "B"]:
            diff = df[f"{comp}_diff"]
            mae = _component_mae(diff)
            f.write(f"{comp} MAE: {mae:.4f}\n")


def compute_crude_mae(df: pd.DataFrame) -> float:
    """Return mean MAE across vector components Bx/By/Bz.

    Args:
        df: Evaluation dataframe containing `Bx_diff`, `By_diff`, `Bz_diff`.

    Returns:
        Scalar MAE summary.
    """
    return float(np.mean([
        np.mean(np.abs(df["Bx_diff"])),
        np.mean(np.abs(df["By_diff"])),
        np.mean(np.abs(df["Bz_diff"])),
    ]))
