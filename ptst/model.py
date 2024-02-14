# Standard
import os
import sys
from glob import glob
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

import random

# Third Party
from transformers import (
    EarlyStoppingCallback,
    PatchTSMixerConfig,
    PatchTSMixerForPrediction,
    Trainer,
    TrainingArguments,
)
import numpy as np
import pandas as pd
import torch

# First Party
from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index

from transformers import set_seed

set_seed(42)

dataset_path = glob("data/raw/*.parquet")
timestamp_column = "STCK_CNTG_HOUR"
id_columns = []

context_length = 512 # 512
forecast_horizon = 96 # 96
num_workers = 16  # Reduce this if you have low number of CPU cores
batch_size = 64  # Adjust according to GPU memory

from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore, boxcox
scaler = StandardScaler()

data = pd.read_parquet(
    dataset_path[11],
    # parse_dates=[timestamp_column],
)
######################
from gluonts.time_feature import get_lags_for_frequency, time_features_from_frequency_str

data["AMOUNT"] = data["STCK_PRPR"] * data["CNTG_VOL"]
data["target"] = data["STCK_PRPR"].pct_change(periods=context_length).fillna(0.)
data["STCK_PRPR"] = scaler.fit_transform(data["STCK_PRPR"].values.reshape(-1,1))
freq = "1s"
lags_sequence = get_lags_for_frequency(freq)
time_features = time_features_from_frequency_str(freq)

timestamp_as_index = pd.DatetimeIndex(data[timestamp_column])
additional_features = [
    (time_feature.__name__, time_feature(timestamp_as_index))
    for time_feature in time_features
]
data = pd.concat([data, pd.DataFrame(dict(additional_features))], axis=1)
data[["AMOUNT", "CNTG_VOL"]] = scaler.fit_transform(data[["AMOUNT", "CNTG_VOL"]])
######################

forecast_columns = list(data.columns.difference([timestamp_column, "MKSC_SHRN_ISCD", "target", "day_of_year", "STCK_PRPR"]))
forecast_columns