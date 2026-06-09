import torch
import torch.nn as nn

def monitoring_loss(y_pred, y_true):
    """Compute plain MAE for monitoring component-wise errors."""

    error = torch.abs(y_pred - y_true)
    return error.mean()


def radial_weights(R, R_max=5, eps=1e-8):
    """Compute radial decay weights.

    Args:
        R: Radius tensor of shape (N,).
        R_max: Maximum radius used for normalization.
        eps: Numerical stability term.

    Returns:
        Tensor of shape (N,) with values clamped to [0, 1].
    """
    if R_max is None:
        R_max = R.max()
    w = 1.0 - R / (R_max + eps)
    # Clamp forces weights between 0 and 1.
    return torch.clamp(w, 0.0, 1.0)


def parallel_perp_loss(y_pred, y_true, BJ, R, w_par=0.5, w_perp=1):
    """Loss on parallel/perpendicular error relative to background field.

    Args:
        y_pred: Predicted magnetic field tensor, shape (N, 3).
        y_true: Target magnetic field tensor, shape (N, 3).
        BJ: Background field tensor, shape (N, 3).
        R: Radius tensor, shape (N,).
        w_par: Weight on parallel error.
        w_perp: Weight on perpendicular error.

    Returns:
        Scalar tensor with weighted mean parallel/perpendicular error.
    """
    # Direction of background field B_J.
    b_hat = BJ / (torch.norm(BJ, dim=1, keepdim=True) + 1e-8)

    dB = y_pred - y_true

    # Parallel component
    dB_par = (dB * b_hat).sum(dim=1, keepdim=True) * b_hat
    dB_perp = dB - dB_par

    err_par = torch.abs(dB_par).sum(dim=1)
    err_perp = torch.abs(dB_perp).sum(dim=1)

    weights = radial_weights(R)

    return torch.mean(weights * (w_par * err_par + w_perp * err_perp))


def calculate_div_B(B_pred, coords):
    """Approximate divergence-like penalty along a trajectory.

    Args:
        B_pred: Predicted magnetic field tensor, shape (N, 3).
        coords: Position tensor, shape (N, 3).

    Returns:
        Scalar tensor: mean absolute gradient proxy along the trajectory.
    """
    # Change in B and Change in Position between points
    dB = B_pred[1:] - B_pred[:-1]
    # We take the norm of the step dR (the distance between points)
    dR = torch.norm(coords[1:] - coords[:-1], dim=1, keepdim=True) + 1e-8

    # Proxy: sum of (component change / distance change)
    # This represents the "gradient" along the trajectory
    if dB.shape[0] == 0:
        return torch.zeros((), device=B_pred.device)
    div_proxy = torch.abs(torch.sum(dB / dR, dim=1))
    return torch.mean(div_proxy)


def total_science_loss_B(B_pred, B_true, BJ, R, coords, w_par=0.5, w_perp=1.0, w_div=0.1):
    """Combined science loss = target error + divergence penalty.

    Args:
        B_pred: Predicted magnetic field tensor, shape (N, 3).
        B_true: Target magnetic field tensor, shape (N, 3).
        BJ: Background field tensor, shape (N, 3).
        R: Radius tensor, shape (N,).
        coords: Position tensor, shape (N, 3).
        w_par: Weight on parallel error term.
        w_perp: Weight on perpendicular error term.
        w_div: Weight on divergence penalty.

    Returns:
        Scalar loss tensor.
    """
    # 1. Target Loss (Your Parallel/Perpendicular logic)
    L_target = parallel_perp_loss(B_pred, B_true, BJ, R, w_par=w_par, w_perp=w_perp)

    # 2. Physics Loss (Divergence proxy)
    L_div = calculate_div_B(B_pred, coords)

    return L_target + (w_div * L_div)


class ScienceLoss(nn.Module):
    """Torch module wrapper around `total_science_loss_B`.

    Args:
        w_par: Weight on parallel error term.
        w_perp: Weight on perpendicular error term.
        w_div: Weight on divergence penalty term.
    """

    def __init__(self, w_par=0.5, w_perp=1.0, w_div=0.1):
        super().__init__()
        self.w_par = w_par
        self.w_perp = w_perp
        self.w_div = w_div

    def forward(self, y_pred, y_true, BJ, R, coords):
        """Compute science loss.

        Args:
            y_pred: Predicted magnetic field tensor, shape (N, 3).
            y_true: Target magnetic field tensor, shape (N, 3).
            BJ: Background field tensor, shape (N, 3).
            R: Radius tensor, shape (N,).
            coords: Position tensor, shape (N, 3).

        Returns:
            Scalar loss tensor.
        """
        return total_science_loss_B(
            y_pred,
            y_true,
            BJ,
            R,
            coords,
            w_par=self.w_par,
            w_perp=self.w_perp,
            w_div=self.w_div,
        )
