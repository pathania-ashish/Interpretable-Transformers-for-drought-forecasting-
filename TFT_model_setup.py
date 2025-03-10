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
test_df=pd.read_csv("__path_to_test_data__")

# The tabular dataset is structured with the following columns:
# 'Date', 'Station_No', 'Step_per_station', 'lat', 'lon', 'Precipitation',
# 'NINO3', 'NINO4', 'NINO3_4', 'PDO', 'IOD', 'SOI', 'PET', 'Tmax', 'Tmin',
# 'Elevation', and 'SPEI_3'.

# Explanation of columns:
# - 'SPEI_3': Target variable representing the Standardized Precipitation Evapotranspiration Index at a 3-month scale.
# - 'Station_No': Unique identifier for each meteorological station.
# - 'Step_per_station': Time step for each station, representing different time periods in the dataset.
# - 'lat', 'lon', 'Elevation': Geographical coordinates and altitude of each station.
# - 'Precipitation', 'PET', 'Tmax', 'Tmin': Regional climate variables including precipitation, potential evapotranspiration, and temperature extremes (maximum and minimum).
# - 'NINO3', 'NINO4', 'NINO3_4', 'PDO', 'IOD', 'SOI': Large-scale climate indices capturing global climate variability such as El Niño and the Indian Ocean Dipole.

# Data types setup:
# Ensure appropriate data types are used for each column to facilitate accurate analysis and modeling.

def convert_dtypes(df):
    df["Station_No"] = df["Station_No"].astype("category")
    df["Step_per_station"] = df["Step_per_station"].astype("int32")
    df["SPEI_3"] = df["SPEI_3"].astype("float64")
    df[['lat', 'lon', "Elevation"]] = df[['lat', 'lon', "Elevation"]].astype("int64")
    df[['Precipitation', 'NINO3', 'NINO4', 'NINO3_4', 'PDO', 'IOD', 'SOI', 'PET', 'Tmax', 'Tmin']] = df[['Precipitation', 'NINO3', 'NINO4', 'NINO3_4', 'PDO', 'IOD', 'SOI', 'PET', 'Tmax', 'Tmin']].astype("float64")
    return df

train_df = convert_dtypes(train_df)
val_df = convert_dtypes(val_df)
test_df = convert_dtypes(test_df)

# Setting up the TimeSeriesDataSet
training = TimeSeriesDataSet(
    train_df,
    time_idx="Step_per_station",
    target="SPEI_3",
    group_ids=["Station_No"],
    min_encoder_length=6, # setting up encoder length
    max_encoder_length=24,   
    min_prediction_length=1, # at a lead time of 1,2,3 months
    max_prediction_length=3,
    static_reals=['lat', 'lon','Elevation'],
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
testing = training.from_dataset(training, test_df, stop_randomization=True, predict=False)
# create dataloaders for  our model
batch_size = 128
# if you have a strong GPU, feel free to increase the number of workers  
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size*10, num_workers=0)

test_dataloader = testing.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

## Setting up the model

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-6, patience=5, verbose=True, mode="min")
lr_logger = LearningRateMonitor()  
logger = TensorBoardLogger("lightning_logs")  
#to save the model
checkpoint_callback = ModelCheckpoint(dirpath='saved_model', filename='model-{epoch:02d}-{val_loss:.2f}', save_top_k=1, monitor='val_loss', mode='min')

trainer = pl.Trainer(
    max_epochs=45,
    accelerator='gpu', 
    devices=1,
    enable_model_summary=True,
    gradient_clip_val=0.2,
    callbacks=[lr_logger, early_stop_callback,checkpoint_callback],
    logger=logger,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    # Enter the hyperparameters optimized using Optuna and further refined through experimentation
    learning_rate=0.00001,
    hidden_size=256,
    attention_head_size=3, 
    dropout=0.2,
    hidden_continuous_size=128,
    output_size=7,  # there are 7 quantiles by default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    loss=QuantileLoss(),
    log_interval=10, 
    reduce_on_plateau_patience=5,
)

trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

#Saving model at checkpoint
checkpoint_path = 'saved_model\model-epoch...path_to_checkpoint..ckpt' #put checkpoint from saved_model above 
torch.save(tft.state_dict(), checkpoint_path)

# Loading the saved model
best_tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.00001,
    hidden_size=256,
    attention_head_size=3, 
    dropout=0.2,
    hidden_continuous_size=128,
    output_size=7,  # there are 7 quantiles by default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    loss=QuantileLoss(),
    log_interval=10, 
    reduce_on_plateau_patience=5,
)

best_tft.load_state_dict(torch.load(checkpoint_path))


# Evaluating performance on validation data
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)  
actuals = actuals.to('cuda:0')
predictions = predictions.to('cuda:0')

# Calculate and print the average p50 loss overall
average_p50_loss = (actuals - predictions).abs().mean().item()
print(f"Average p50 validation loss overall: {average_p50_loss}")


# Evaluating performance on test data
actuals = torch.cat([y[0] for x, y in iter(test_dataloader)])
predictions = best_tft.predict(test_dataloader)

actuals = actuals.to('cuda:0')
predictions = predictions.to('cuda:0')

# Calculate and print the average p50 loss overall
average_p50_loss = (actuals - predictions).abs().mean().item()
print(f"Average p50 testing loss overall: {average_p50_loss}")

actuals1 = actuals.to('cpu')
predictions1 = predictions.to('cpu')

# ### Testing Dataset Results

# Visualizing the predictions
raw_predictions = best_tft.predict(test_dataloader, mode="raw", return_x=True)

# Transfer tensors from GPU to CPU
actuals_cpu = actuals.cpu()
predictions_cpu = predictions.cpu()
actuals_100 = actuals_cpu[0:100]
predictions_100 = predictions_cpu[0:100]

# visualizing the first 100 predictions for a station
plt.figure()
plt.plot(actuals_100[:, 0], label='Actuals')
plt.plot(predictions_100[:, 0], label='Predictions')
plt.title('Prediction at 30 days ahead')
plt.legend()
plt.show()

# obtaining the quantile predictions for the test data
predict_q= best_tft.predict(test_dataloader, mode="quantiles", num_workers=3) 
