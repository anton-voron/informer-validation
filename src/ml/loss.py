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

    raise ValueError("Unknown loss")


class GMADL(MultiHorizonMetric):
    """
    GMADL (Generalized Mean Absolute Difference Loss)

    loss = - (σ(a * ŷ * y) - 0.5) * |y|^b
    where σ(z) = 1 / (1 + exp(−z)), a controls sigmoid steepness,
    and b controls the weighting by the magnitude of the target.
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

        # final loss: negative sign to turn into a minimization objective
        loss_tensor = - sigmoid_term * weight

        return loss_tensor
        
        # return -1 * \
        #     (1 / (1 + torch.exp(-self.a * self.to_prediction(y_pred) * target)
        #           ) - 0.5) * torch.pow(torch.abs(target), self.b)

