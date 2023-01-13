import pandas as pd
from utils import *
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from utils.Factory import SequenceDataset, ShallowRegressionLSTM

# First, we read in the data, dropping the index and the date.

df = pd.read_csv('C:\\Users\\Aarus\\Documents\\code\\HurricanePredictionUsingLSTM\\static\\dataset\\atlantic (2).csv')

df.drop(['status_of_system', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8','Unnamed: 9', 
        'Unnamed: 10', 'Unnamed: 11','Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 
        'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18'], inplace=True, axis=1)
df['date'] = pd.to_datetime(df['date'], format = "%Y%m%d").dt.strftime('%Y-%m-%d')


df['tmrw windspeed'] =  df['max_sustained_wind'].shift(-1)
df['date'] = df['date'].apply(lambda x: float(x.split()[0].replace('-', '')))
df['latitude'] = df['latitude'].map(lambda x: float(x.rstrip('NEWS')))
df['longitude'] = df['longitude'].map(lambda x: float(x.rstrip('NEWS')))
df['max_sustained_wind'] = df['max_sustained_wind'].replace(-99, np.nan)

df['max_sustained_wind'] =  df['max_sustained_wind'].fillna(method='ffill', limit=1000)
df['max_sustained_wind'] =  df['max_sustained_wind'].fillna(method='bfill', limit=1000)
df['central_pressure'] = df['central_pressure'].fillna(method='ffill', limit=1000)
df['central_pressure'] = df['central_pressure'].fillna(method='bfill', limit=1000)
df['tmrw windspeed'] = df['tmrw windspeed'].fillna(method='ffill', limit=1000)
df['tmrw windspeed'] = df['tmrw windspeed'].fillna(method='bfill', limit=1000)


target = "tmrw windspeed"
features = list(df.columns.difference(["date", 'tmrw windspeed']))


# Data Processing
# To process the data, we first split it into training and test data, where two-thirds of the data is used for training, and the last third is used for testing.

size = int(len(df) * 0.7)
df_train = df.loc[:size].copy()
df_test = df.loc[size:].copy()


# Next, in order to ensure that some values due to their mangnitude do not inherently dominate the features, we standardize their values.
target_mean = df_train[target].mean()
target_stdev = df_train[target].std()

for c in df_train.columns:
    mean = df_train[c].mean()
    stdev = df_train[c].std()

    df_train[c] = (df_train[c] - mean) / stdev
    df_test[c] = (df_test[c] - mean) / stdev


# Finally, the last step in the data processing to prepare for LSTM is to prepare the data in a sequence of past observations. Preparation of the LSTM on time series data means that it uses a certain number of past observations to predict the future. In this case, the sequence length decides how many days the LSTM considers in advance. If the sequence length is $n$, then the LSTM considers the last $n$ observations to predict the $n+1$th day.
# We decided the sequence length as 3 for purposes of this notebook.

torch.manual_seed(101)

batch_size = 1
sequence_length = 3

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


# Running the Classical LSTM
learning_rate = 0.0001
num_hidden_units = 16

model = ShallowRegressionLSTM(num_sensors=len(features), hidden_units=num_hidden_units)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


classical_loss_train = []
classical_loss_test = []
print("Untrained test\n--------")
test_loss = test_model(test_loader, model, loss_function)
print()
classical_loss_test.append(test_loss)

for ix_epoch in range(20):
    print(f"Epoch {ix_epoch}\n---------")
    train_loss = train_model(train_loader, model, loss_function, optimizer=optimizer)
    test_loss = test_model(test_loader, model, loss_function)
    print()
    classical_loss_train.append(train_loss)
    classical_loss_test.append(test_loss)

# Predict
def predict(data_loader, model):
    """Just like `test_loop` function but keep track of the outputs instead of the loss
    function.
    """
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)
    
    return output

train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

ystar_col = "Model forecast"
df_train[ystar_col] = predict(train_eval_loader, model).numpy()
df_test[ystar_col] = predict(test_loader, model).numpy()

df_out = pd.concat((df_train, df_test))[[target, ystar_col]]

for c in df_out.columns:
    df_out[c] = df_out[c] * target_stdev + target_mean
