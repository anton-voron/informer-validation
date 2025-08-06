import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__),  '..'))


import hydra
from omegaconf import DictConfig, OmegaConf


import logging
import pprint
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tempfile

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score




from ml.data import build_time_series_dataset
from ml.loss import get_loss
from ml.model import get_model




@hydra.main(version_base=None, config_path="../config", config_name="informer")
def train(cfg : DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    conf = OmegaConf.to_container(cfg, resolve=True)

    pl.seed_everything(conf['experiment']['seed'], workers=True)


    # Get time series dataset
    df = pd.read_csv(conf['data']['path'])
    df['weekday'] = df['weekday'].astype(str)
    df['hour'] = df['hour'].astype(str)
    
    for col in ["group_ids", "weekday", "hour"]:
        df[col] = df[col].astype("category")

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
    # logging.info(f"Model hyperparameters: \n{pprint.pformat(model.hparams)}")
    
    
    # Callbacks
    
    logger = CSVLogger(
        save_dir=conf['experiment']['log_dir'],
        name=conf['experiment']['name'],
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=conf['experiment']['early_stopping_patience'],
        mode="min",
        stopping_threshold=0.001
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        monitor="val_loss",
        verbose=True,
        # save_last=False,
        save_top_k=5,
        # every_n_epochs=2,
        mode="min",
    )
    
    batch_size = conf['batch_size']
    max_epochs = conf['max_epochs']
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback],
        log_every_n_steps=conf['experiment']['log_interval'],
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    trainer.fit(
        model=model,
        train_dataloaders=train_dataset.to_dataloader(batch_size=batch_size, shuffle=True),
        val_dataloaders=val_dataset.to_dataloader(batch_size=batch_size, shuffle=False),
    )
    
    logging.info("Training complete.")
    logging.info(f"Best model saved at: {checkpoint_callback.best_model_path}")
    logging.info(f"Best model score: {checkpoint_callback.best_model_score}")
    
    # Test evaluation on the full test dataset
    logging.info("Starting test evaluation...")
    
    # Load the best model for testing
    if checkpoint_callback.best_model_path:
        best_model = model.__class__.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            loss=loss
        )
        logging.info(f"Loaded best model from: {checkpoint_callback.best_model_path}")
    else:
        best_model = model
        logging.info("Using final model for testing (no checkpoint found)")
    
    # Run test evaluation
    test_results = trainer.test(
        model=best_model,
        dataloaders=test_dataset.to_dataloader(batch_size=batch_size, shuffle=False),
        verbose=True
    )
    
    logging.info("Test evaluation completed.")
    
    # Parse and display test results more clearly
    if test_results and len(test_results) > 0:
        test_metrics = test_results[0]
        logging.info("=" * 50)
        logging.info("TEST RESULTS SUMMARY:")
        logging.info("=" * 50)
        
        # Extract and display key metrics
        test_loss = test_metrics.get('test_loss', 'N/A')
        test_mae = test_metrics.get('test_MAE', 'N/A')
        test_rmse = test_metrics.get('test_RMSE', 'N/A')
        
        # Look for R2 score in different possible keys
        test_r2 = None
        for key, value in test_metrics.items():
            if 'r2' in key.lower() or 'R2Score' in key or 'TorchMetricWrapper' in key:
                test_r2 = value
                break
        
        logging.info(f"Test Loss: {test_loss}")
        logging.info(f"Test MAE (Mean Absolute Error): {test_mae}")
        logging.info(f"Test RMSE (Root Mean Squared Error): {test_rmse}")
        logging.info(f"Test R2 Score: {test_r2 if test_r2 is not None else 'N/A'}")
        logging.info("=" * 50)
    
    logging.info(f"Full test results: {test_results}")
    logging.info(f"Training logs saved at: {logger.log_dir}")
    logging.info(f"Training configuration: \n{pprint.pformat(conf)}")
    logging.info("Training finished successfully.")
    

    
    


if __name__ == "__main__":
    train()
