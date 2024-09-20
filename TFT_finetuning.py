"""
@author: Ashish Pathania
"""
# importing the necessary libraries
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf 
import tensorboard as tb 
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from lightning.pytorch.callbacks import ModelCheckpoint

# reading the train, validation and test data
train_df=pd.read_csv("__path_to_train_data__")
val_df=pd.read_csv("__path_to_val_data__")


def convert_dtypes(df):
    df["Station_No"] = df["Station_No"].astype("category")
    df["Step_per_station"] = df["Step_per_station"].astype("int32")
    df["SPEI_3"] = df["SPEI_3"].astype("float64")
    df[['lat', 'lon', "Elevation"]] = df[['lat', 'lon', "Elevation"]].astype("int64")
    df[['Precipitation', 'NINO3', 'NINO4', 'NINO3_4', 'PDO', 'IOD', 'SOI', 'PET', 'Tmax', 'Tmin']] = df[['Precipitation', 'NINO3', 'NINO4', 'NINO3_4', 'PDO', 'IOD', 'SOI', 'PET', 'Tmax', 'Tmin']].astype("float64")
    return df

train_df = convert_dtypes(train_df)
val_df = convert_dtypes(val_df)


# Setting up the TimeSeriesDataSet
training = TimeSeriesDataSet(
    train_df,
    time_idx="Step_per_station",
    target="SPEI_3",
    group_ids=["Station_No"],
    min_encoder_length=24, # setting up encoder length
    max_encoder_length=24,   
    min_prediction_length=3, # at a lead time of 1,2,3 months
    max_prediction_length=3,
    static_reals=['lat', 'lon',"Elevation"],
    time_varying_known_reals=['Step_per_station'], 
    time_varying_unknown_reals=["SPEI_3",'Precipitation','NINO3', 'NINO4', 'NINO3_4', 'PDO', 'IOD', 'SOI', 'PET', 'Tmax', 'Tmin'],
    target_normalizer=GroupNormalizer(
        groups=["Station_No"], transformation=None
    ),  # we normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    
) 

validation = training.from_dataset(training, val_df, stop_randomization=True, predict=True)
# create dataloaders for  our model
batch_size = 128
# if you have a strong GPU, feel free to increase the number of workers  
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size*10, num_workers=0)

import pickle

from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

# create study
study = optimize_hyperparameters(
    train_dataloader,
    val_dataloader,
    model_path="optuna_test",
    n_trials=200,
    max_epochs=50,
    gradient_clip_val_range=(0.01, 100),
    hidden_size_range=(8, 512),
    hidden_continuous_size_range=(8, 512),
    attention_head_size_range=(1, 6),
    learning_rate_range=(0.000001, 0.1),
    dropout_range=(0.1, 0.4),
    trainer_kwargs=dict(limit_train_batches=30),
    reduce_on_plateau_patience=4,
    use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
)

# save study results - also we can resume tuning at a later point in time
with open("test_study.pkl", "wb") as fout:
    pickle.dump(study, fout)

# show best hyperparameters
print(study.best_trial.params)
## Setting up the model
