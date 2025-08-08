import torch
from torch import Tensor
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric

class GMADL(MultiHorizonMetric):
    r"""
    Generalized Mean Absolute Directional Loss (GMADL)

    Per-element loss:
        L_i = sigmoid(-a * y_hat_i * y_i) * (|y_i|^b_clipped)

    where:
      - y_hat = model prediction in target scale (via `to_prediction`)
      - a > 0 controls the steepness of the directional reward (bigger a → stronger sign emphasis)
      - b >= 0 controls how much large-magnitude targets are emphasized
      - we use sigmoid(-z) instead of (1 - sigmoid(z)) for numerical stability

    Notes / good defaults:
      * If you train on volatility-normalized returns, start with a≈20, b≈1.
      * If you train on raw returns, consider pre-winsorizing or use a smaller b and set `weight_clip`.
      * This class returns an element-wise tensor; reduction is handled by PyTorch Forecasting.
    """

    def __init__(
        self,
        a: float = 20.0,
        b: float = 1.0,
        eps: float = 1e-12,
        weight_clip: float | None = None,
        **kwargs,
    ):
        """
        Args:
            a: sigmoid steepness for directional term (too large saturates gradients).
            b: exponent on |target| for magnitude weighting (b=0 disables magnitude emphasis).
            eps: small constant to keep weights finite (|y|^b on tiny |y|).
            weight_clip: optional upper bound for the magnitude weight to avoid exploding loss.
        """
        super().__init__(**kwargs)
        self.a = float(a)
        self.b = float(b)
        self.eps = float(eps)
        self.weight_clip = None if weight_clip is None else float(weight_clip)

    def loss(self, y_pred: Tensor, target: Tensor) -> Tensor:
        # predictions in target scale and broadcasted to target shape
        y_hat = self.to_prediction(y_pred)
        y = target

        # Directional term: sigmoid(-a * y_hat * y) \in (0, 1)
        # If signs agree and |y_hat*y| is large, term → small → small loss
        z = -self.a * (y_hat * y)
        dir_term = torch.sigmoid(z)

        # Magnitude weight: (|y| + eps)^b, with optional clipping
        weight = torch.clamp(torch.abs(y), min=self.eps).pow(self.b)
        if self.weight_clip is not None:
            weight = torch.clamp(weight, max=self.weight_clip)

        # Element-wise loss (no reduction here)
        return dir_term * weight
