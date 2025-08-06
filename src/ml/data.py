import os

import pandas as pd
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet





def build_time_series_dataset(config, data) -> TimeSeriesDataSet:
    data = data.copy()
 
    time_series_dataset = TimeSeriesDataSet(
        data,
        time_idx=config['data']['time_index'],
        target=config['data']['target'],
        group_ids=config['data']['group_ids'],
        min_encoder_length=config['past_window'],
        max_encoder_length=config['past_window'],
        min_prediction_length=config['future_window'],
        max_prediction_length=config['future_window'],
        static_reals=config['data']['static_real'],
        static_categoricals=config['data']['static_cat'],
        time_varying_known_reals=config['data']['dynamic_known_real'],
        time_varying_known_categoricals=config['data']['dynamic_known_cat'],
        time_varying_unknown_reals=config['data']['dynamic_unknown_real'],
        time_varying_unknown_categoricals=config['data']['dynamic_unknown_cat'],
        randomize_length=False,
    )

    return time_series_dataset
