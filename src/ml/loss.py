import torch
from torch import Tensor

from pytorch_forecasting import QuantileLoss, RMSE
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric


def get_loss(config):
    loss_name = config['loss']['name']

    if loss_name == 'Quantile':
        return QuantileLoss(config['loss']['quantiles'])

    if loss_name == 'GMADL':
        return GMADL(
            a=config['loss']['a'],
            b=config['loss']['b']
        )
    
    if loss_name == 'RMSE':
        return RMSE()
    
    if loss_name == 'R2Loss':
        return R2Loss()

    raise ValueError("Unknown loss")


class GMADL(MultiHorizonMetric):
    """
    GMADL (Generalized Mean Absolute Difference Loss)

    loss = (0.5 - σ(a * ŷ * y)) * |y|^b
    where σ(z) = 1 / (1 + exp(−z)), a controls sigmoid steepness,
    and b controls the weighting by the magnitude of the target.
    
    This loss encourages predictions that have the same sign as the target.
    When ŷ and y have the same sign, the loss decreases.
    """

    def __init__(self, a: float = 1000.0, b: float = 2.0, **kwargs):
        """
        Args:
            a: scaling factor inside the sigmoid
            b: exponent on |target| for weighting
        """
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def loss(self, y_pred: Tensor, target: Tensor) -> Tensor:
        # get the model’s predictions in the same scale as target
        # (MultiHorizonMetric handles any necessary reshaping internally)
        y_hat = self.to_prediction(y_pred)

        # scaled product inside the sigmoid
        z = self.a * y_hat * target

        # sigmoid term in brackets: σ(z) - 0.5
        sigmoid_term = torch.sigmoid(z) - 0.5

        # magnitude-based weighting: |target|^b
        weight = torch.abs(target).pow(self.b)

        # final loss: convert to positive loss for minimization
        # when predictions and targets have same sign, sigmoid_term > 0, so we want to minimize (1 - sigmoid_term)
        loss_tensor = (0.5 - sigmoid_term) * weight

        return loss_tensor
        
        # return -1 * \
        #     (1 / (1 + torch.exp(-self.a * self.to_prediction(y_pred) * target)
        #           ) - 0.5) * torch.pow(torch.abs(target), self.b)


class R2Loss(MultiHorizonMetric):
    """
    R2Loss (R-squared Loss)
    
    Converts R² coefficient of determination into a loss function.
    Loss = 1 - R²
    
    R² = 1 - (SS_res / SS_tot)
    where:
    - SS_res = Σ(y_true - y_pred)²  (residual sum of squares)
    - SS_tot = Σ(y_true - y_mean)²  (total sum of squares)
    
    Perfect predictions give R² = 1, so loss = 0
    Random predictions give R² ≈ 0, so loss ≈ 1
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def loss(self, y_pred: Tensor, target: Tensor) -> Tensor:
        # get the model's predictions in the same scale as target
        y_hat = self.to_prediction(y_pred)
        
        # Calculate mean of target values
        y_mean = torch.mean(target)
        
        # Calculate sum of squares
        ss_res = torch.sum((target - y_hat) ** 2)  # residual sum of squares
        ss_tot = torch.sum((target - y_mean) ** 2)  # total sum of squares
        
        # Calculate R²
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        r2 = 1 - (ss_res / (ss_tot + eps))
        
        # Convert R² to loss: loss = 1 - R²
        # Clamp to ensure loss is non-negative
        loss_tensor = torch.clamp(1 - r2, min=0.0)
        
        return loss_tensor

