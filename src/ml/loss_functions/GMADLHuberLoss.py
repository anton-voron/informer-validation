import torch
from torch import Tensor
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric

class GMADLHuberLoss(MultiHorizonMetric):
    """
    Train on volatility-normalized returns (target ~ N(0,1)-ish).
    loss_i = w_dir * (0.5 - sigmoid(a * y_hat * y)) * |y|^b  +  w_reg * huber(y_hat - y; delta)
    - Direction term (GMADL-like) rewards correct sign and larger magnitudes.
    - Robust regression term (Huber) prevents collapse and handles fat tails.
    """
    def __init__(self, a: float = 20.0, b: float = 1.0, delta: float = 1.0,
                 w_dir: float = 1.0, w_reg: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b
        self.delta = delta
        self.w_dir = w_dir
        self.w_reg = w_reg

    @staticmethod
    def _huber(residual: Tensor, delta: float) -> Tensor:
        abs_r = residual.abs()
        quad = 0.5 * (abs_r**2)
        lin  = delta * (abs_r - 0.5*delta)
        return torch.where(abs_r <= delta, quad, lin)

    def loss(self, y_pred: Tensor, target: Tensor) -> Tensor:
        # y_hat has same shape as target after to_prediction
        y_hat = self.to_prediction(y_pred)
        y = target

        # Directional (GMADL-like)
        dir_term = (0.5 - torch.sigmoid(self.a * y_hat * y)) * torch.abs(y).pow(self.b)

        # Robust regression on vol-normalized residuals
        reg_term = self._huber(y_hat - y, self.delta)

        return self.w_dir * dir_term + self.w_reg * reg_term
