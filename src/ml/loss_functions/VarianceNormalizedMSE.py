import torch
from torch import Tensor
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric

class VarianceNormalizedMSE(MultiHorizonMetric):
    r"""
    Per-element variance-normalized MSE suitable for multi-horizon forecasting.

    For each sample (and channel), compute the baseline variance over the horizon:
        var_ref = mean_h[(y - mean_h(y))^2]

    Then normalize each squared error by var_ref:
        loss_ij = (y_ij - yhat_ij)^2 / (var_ref_i + eps)

    Summing/averaging over the horizon approximates a per-sample (1 - R^2) style objective
    while keeping gradients local and stable. (R² background & properties: scikit-learn docs.)  # noqa
    """
    def __init__(self, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = float(eps)

    def loss(self, y_pred: Tensor, target: Tensor) -> Tensor:
        y_hat = self.to_prediction(y_pred)
        y = target

        # Ensure shape (B, H[, C])
        if y.dim() == 1:
            y = y.unsqueeze(0)
            y_hat = y_hat.unsqueeze(0)

        horizon_dim = 1  # PyTorch Forecasting convention: time/horizon axis = 1

        # Per-sample (and per-channel) horizon mean & variance
        y_mean = y.mean(dim=horizon_dim, keepdim=True)
        var_ref = (y - y_mean).pow(2).mean(dim=horizon_dim, keepdim=True)

        # Element-wise normalized squared error
        se = (y - y_hat).pow(2)
        loss_elem = se / (var_ref + self.eps)
        return loss_elem
