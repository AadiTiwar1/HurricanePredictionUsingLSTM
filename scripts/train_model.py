import pandas as pd
from utils import *

import time
import numpy as np
import math
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch import nn


df = pd.read_csv('../static/dataset/atlantic (2).csv')

df.drop(['status_of_system', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8','Unnamed: 9', 
        'Unnamed: 10', 'Unnamed: 11','Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 
        'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18'], inplace=True, axis=1)
df['date'] = pd.to_datetime(df['date'], format = "%Y%m%d").dt.strftime('%Y-%m-%d')


df['next_windspeed'] =  df['max_sustained_wind'].shift(-1)

# df.to_csv('cleanedHurricaneData.csv', index=False)

df['date'] = df['date'].apply(lambda x: float(x.split()[0].replace('-', '')))
df['latitude'] = df['latitude'].map(lambda x: float(x.rstrip('NEWS')))
df['longitude'] = df['longitude'].map(lambda x: float(x.rstrip('NEWS')))
df['max_sustained_wind'] = df['max_sustained_wind'].replace(-99, np.nan)
df['next_windspeed'] = df['next_windspeed'].replace(-99, np.nan)

df['max_sustained_wind'] =  df['max_sustained_wind'].fillna(method='ffill', limit=1000)
df['max_sustained_wind'] =  df['max_sustained_wind'].fillna(method='bfill', limit=1000)
df['central_pressure'] = df['central_pressure'].fillna(method='ffill', limit=1000)
df['central_pressure'] = df['central_pressure'].fillna(method='bfill', limit=1000)
df['next_windspeed'] = df['next_windspeed'].fillna(method='ffill', limit=1000)
df['next_windspeed'] = df['next_windspeed'].fillna(method='bfill', limit=1000)

target = "next_windspeed"
features = list(df.columns.difference([target]))
size = int(len(df) * 0.8)

df_train = df.loc[:size].copy()
df_test = df.loc[size:].copy()
target_mean = df_train[target].mean()
target_stdev = df_train[target].std()

for c in df_train.columns:
    mean = df_train[c].mean()
    stdev = df_train[c].std()

    df_train[c] = (df_train[c] - mean) / stdev
    df_test[c] = (df_test[c] - mean) / stdev
from utils.Factory import SequenceDataset
torch.manual_seed(345089723)

batch_size = 1
sequence_length = 6

train_dataset = SequenceDataset(
    df_train,
    target=target,
    features=features,
    sequence_length=sequence_length
)
test_dataset = SequenceDataset(
    df_test,
    target=target,
    features=features,
    sequence_length=sequence_length
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

X, y = next(iter(train_loader))

print("Features shape:", X.shape)
print("Target shape:", y.shape)
def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    
    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")
    return avg_loss

def test_model(data_loader, model, loss_function):
    
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")
    return avg_loss
from utils.Factory import ShallowRegressionLSTM
learning_rate = 0.0001
num_hidden_units = 16

model = ShallowRegressionLSTM(num_sensors=len(features), hidden_units=num_hidden_units)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
