from pytorch_forecasting import QuantileLoss, RMSE
from .loss_functions import R2Loss, VarianceNormalizedMSE, CombinedLoss, GMADL, GMADLHuberLoss


def get_loss(config):
    loss_name = config['loss']['name']

    if loss_name == 'Quantile':
        return QuantileLoss(config['loss']['quantiles'])

    if loss_name == 'GMADL':
        return GMADL(
            a=config['loss'].get('a', 20.0),
            b=config['loss'].get('b', 1.0),
        )
        
    if loss_name == 'GMADLHuberLoss':
        return GMADLHuberLoss(
            a=config['loss'].get('a', 20.0),
            b=config['loss'].get('b', 1.0),
            delta=config['loss'].get('delta', 1.0),
            w_dir=config['loss'].get('w_dir', 1.0),
            w_reg=config['loss'].get('w_reg', 1.0)
        )
    
    if loss_name == 'RMSE':
        return RMSE()
    
    if loss_name == 'R2Loss':
        return R2Loss()
    
    if loss_name == 'VarianceNormalizedMSE':
        return VarianceNormalizedMSE(eps=config['loss'].get('eps', 1e-8))
    
    if loss_name == 'CombinedLoss':
        return CombinedLoss(
            mse_weight=config['loss'].get('mse_weight', 0.7),
            norm_weight=config['loss'].get('norm_weight', 0.3),
            eps=config['loss'].get('eps', 1e-8)
        )

    raise ValueError("Unknown loss")


