import os
from typing import Union

from torchmetrics import R2Score
from pytorch_forecasting.metrics import MAE, RMSE, QuantileLoss
from pytorch_forecasting.metrics.base_metrics import Metric
from pytorch_forecasting.models.temporal_fusion_transformer import (
    TemporalFusionTransformer)
import torch

from models.informer.model_pytorch_forecasting import InformerPTForecasting


class R2ScoreMetric(Metric):
    """Custom R2Score metric wrapper for pytorch forecasting"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.r2_score = R2Score()
        
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Handle case where y_pred might be a tuple or list
        if isinstance(y_pred, (tuple, list)):
            y_pred = y_pred[0]  # Take the first element (usually the prediction)
        
        # Handle case where y_true might be a tuple or list
        if isinstance(y_true, (tuple, list)):
            y_true = y_true[0]
            
        # Ensure tensors are properly shaped
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)
        
        return self.r2_score(y_pred_flat, y_true_flat)
    
    def __repr__(self) -> str:
        return "R2Score()"



def get_model(config, dataset, loss):
    model_name = config['model']['name']

    if model_name == 'TemporalFusionTransformer':
        return TemporalFusionTransformer.from_dataset(
            dataset,
            hidden_size=config['model']['hidden_size'],
            dropout=config['model']['dropout'],
            attention_head_size=config['model']['attention_head_size'],
            hidden_continuous_size=config['model']['hidden_continuous_size'],
            learning_rate=config['model']['learning_rate'],
            share_single_variable_networks=False,
            loss=loss,
            logging_metrics=[MAE(), RMSE(), R2ScoreMetric()],
        )

    if model_name == 'Informer':
        # Handle categorical encoders safely
        embedding_sizes = {}
        if dataset.categorical_encoders is not None:
            embedding_sizes = {
                name: (len(encoder.classes_), config['model']['d_model'])
                for name, encoder in dataset.categorical_encoders.items()
                if name in dataset.categoricals
            }
        
        return InformerPTForecasting.from_dataset(
            dataset,
            d_model=config['model']['d_model'],
            d_fully_connected=config['model']['d_fully_connected'],
            n_attention_heads=config['model']['n_attention_heads'],
            n_encoder_layers=config['model']['n_encoder_layers'],
            n_decoder_layers=config['model']['n_decoder_layers'],
            dropout=config['model']['dropout'],
            learning_rate=config['model']['learning_rate'],
            loss=loss,
            embedding_sizes=embedding_sizes,
            logging_metrics=[MAE(), RMSE(), R2ScoreMetric()],
        )

    raise ValueError("Unknown model")



def load_model(config) -> Union[InformerPTForecasting, TemporalFusionTransformer]:
    """
    Load a model from the specified path.
    """
    
    model_path = config['model']['path']
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    
    if config['model']['name'] == 'Informer':
        return InformerPTForecasting.load_from_checkpoint(model_path)
    elif config['model']['name'] == 'TemporalFusionTransformer':
        return TemporalFusionTransformer.load_from_checkpoint(model_path)
    
    raise ValueError("Unknown model")
