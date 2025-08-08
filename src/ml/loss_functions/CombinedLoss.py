
import torch
from torch import Tensor
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric

class CombinedLoss(MultiHorizonMetric):
    r"""
    Weighted sum of plain MSE and per-sample variance-normalized MSE.

        loss = w_mse * (y - yhat)^2 + w_norm * (y - yhat)^2 / (var_ref + eps)

    where var_ref is computed per sample (and per channel) over the horizon.
    This keeps the "scale-awareness" of MSE while penalizing errors more when
    the target horizon has low variance (i.e., harder-to-justify deviations).

    Notes:
      * Returning element-wise losses lets PyTorch Forecasting handle masking/reduction.  # noqa
      * Normalization design mirrors NRMSE / (1 - R²) intuition. :contentReference[oaicite:2]{index=2}
    """
    def __init__(self, mse_weight: float = 0.7, norm_weight: float = 0.3, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.mse_weight = float(mse_weight)
        self.norm_weight = float(norm_weight)
        self.eps = float(eps)

    def loss(self, y_pred: Tensor, target: Tensor) -> Tensor:
        y_hat = self.to_prediction(y_pred)
        y = target

        # Ensure shape (B, H[, C])
        if y.dim() == 1:
            y = y.unsqueeze(0)
            y_hat = y_hat.unsqueeze(0)

        horizon_dim = 1

        # Plain MSE per element
        mse = (y - y_hat).pow(2)

        # Variance-normalized term (per sample over horizon)
        y_mean = y.mean(dim=horizon_dim, keepdim=True)
        var_ref = (y - y_mean).pow(2).mean(dim=horizon_dim, keepdim=True)
        norm_mse = mse / (var_ref + self.eps)

        return self.mse_weight * mse + self.norm_weight * norm_mse