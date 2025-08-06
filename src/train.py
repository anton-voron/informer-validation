import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__),  '..'))


import hydra
from omegaconf import DictConfig, OmegaConf


import logging
import pprint
import os
import tempfile
import torch
import lightning.pytorch as pl
import pandas as pd
from sklearn.model_selection import train_test_split

from pytorch_forecasting.data.timeseries import TimeSeriesDataSet

from ml.data import build_time_series_dataset
from ml.loss import get_loss
from ml.model import get_model




@hydra.main(version_base=None, config_path="../config", config_name="informer")
def train(cfg : DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    pl.seed_everything(42, workers=True)
    
    
    # Get time series dataset
    
    conf = OmegaConf.to_container(cfg, resolve=True)
    
    df = pd.read_csv(conf['data']['path'])
    df['weekday'] = df['weekday'].astype(str)
    df['hour'] = df['hour'].astype(str)

    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=False)
    val_df, test_df = train_test_split(val_df, test_size=0.5, shuffle=False)
    logging.info(f"Train shape: {train_df.shape}, Validation shape: {val_df.shape}, Test shape: {test_df.shape}")
    
    train_dataset: TimeSeriesDataSet = build_time_series_dataset(conf, train_df)
    val_dataset: TimeSeriesDataSet = TimeSeriesDataSet.from_dataset(train_dataset, val_df, predict=False)
    test_dataset: TimeSeriesDataSet = TimeSeriesDataSet.from_dataset(train_dataset, test_df, predict=False)
    
    
    # Get loss
    loss = get_loss(conf)
    logging.info(f"Using loss {loss}")

    # Get model
    model = get_model(conf, train_dataset, loss)
    logging.info(f"Using model {model}")

if __name__ == "__main__":
    train()
