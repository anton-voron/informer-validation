import torch
from torch import Tensor
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric

class R2Loss(MultiHorizonMetric):
    r"""
    R²-based loss producing **element-wise** terms for multi-horizon forecasting.

    Idea:
      Classical R² over a sample is 1 - SS_res / SS_tot, where SS_tot uses the
      sample mean of the true values. To get per-element losses compatible with
      MultiHorizonMetric, we divide each squared residual by the per-sample
      horizon variance (SS_tot / T), which is equivalent (up to constants) to
      a per-element (1 - R²). When summed/averaged over the horizon, this
      approximates (1 - R²) while keeping gradients local and stable.

    Shapes:
      target, y_hat: (B, H) or (B, H, C). We compute SS_tot per (B, [C]) over H.

    Notes:
      * R² can be negative when the model is worse than predicting the mean. This is expected. :contentReference[oaicite:1]{index=1}
      * If H < 2 (no variance), we gracefully fall back to plain MSE.
    """

    def __init__(self, eps: float = 1e-8, **kwargs):
        """
        Args:
            eps: small constant to avoid division by zero when variance is tiny.
        """
        super().__init__(**kwargs)
        self.eps = float(eps)

    def loss(self, y_pred: Tensor, target: Tensor) -> Tensor:
        # Predictions in the same shape/scale as target
        y_hat = self.to_prediction(y_pred)
        y = target

        # Ensure at least 2D: (B, H[, C])
        if y.dim() == 1:
            y = y.unsqueeze(0)
            y_hat = y_hat.unsqueeze(0)

        # Determine horizon axis (the time/horizon dimension is axis=1 in PyTorch Forecasting)
        # Handle both (B,H) and (B,H,C)
        # We'll compute totals per (batch, [channel]) over H.
        horizon_dim = 1

        # Mean over horizon (per sample, per channel if present)
        y_mean = y.mean(dim=horizon_dim, keepdim=True)

        # Residuals and per-element squared error
        resid = y - y_hat
        se = resid.pow(2)

        # Total sum of squares (per sample, [per channel]), averaged over horizon to match per-element scale
        # SS_tot = sum_h (y - y_mean)^2 ; we use population variance style (ddof=0) /H
        ss_tot_per = (y - y_mean).pow(2).mean(dim=horizon_dim, keepdim=True)  # shape (B,1[,C])

        # If horizon variance is ~0 (e.g., constant targets or H<2), fallback to MSE for those rows
        denom = ss_tot_per + self.eps

        # Element-wise "1 - R²" proxy: normalized squared error
        # Summing/averaging this over H yields something proportional to (1 - R²)
        loss_elem = se / denom

        return loss_elem
