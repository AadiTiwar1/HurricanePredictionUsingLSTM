#get all of the packages needed
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None
tf.random.set_seed(0)

#download the data and do a little cleanup
def convert_lat_long(value):
    if value[-1] == 'S' or value[-1] == 'W':
        return -1 * float(value[:-1])
    else:
        return float(value[:-1])

df = pd.read_csv('static/dataset/atlantic.csv', sep=',', index_col=False)

df = df.loc[48000:].copy() # we only want to work with a fraction of the data, working with too much will take too long to train

df = df[['date', 'latitude', 'longitude', 'max_sustained_wind', 'central_pressure']]
df['date'] = pd.to_datetime(df['date'], format = "%Y%m%d").dt.strftime('%Y-%m-%d')
df['date'] = df['date'].apply(lambda x: float(x.split()[0].replace('-', '')))

df['latitude'] = df['latitude'].map(lambda x: convert_lat_long(x))
df['longitude'] = df['longitude'].map(lambda x: convert_lat_long(x))

df['central_pressure'] = df['central_pressure'].fillna(method='ffill', limit=1000)
df['central_pressure'] = df['central_pressure'].fillna(method='bfill', limit=1000)
df['max_sustained_wind'] = df['max_sustained_wind'].replace(-99, np.nan)

df['date'] = pd.to_datetime(df['date'], format = "%Y%m%d").dt.strftime('%Y-%m-%d') 
df = df.set_index('date')

y = df['max_sustained_wind'].fillna(method='ffill')
y = y.values.reshape(-1, 1)

# scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(y)
y = scaler.transform(y)

# generate the input and output sequences
n_lookback = 150  # length of input sequences (lookback period)
n_forecast = 500  # length of output sequences (forecast period)

# Generate training sequences and target sequences
# for the LSTM model 
X = []
Y = []

for i in range(n_lookback, len(y) - n_forecast + 1):
    X.append(y[i - n_lookback: i])
    Y.append(y[i: i + n_forecast])

X = np.array(X)
Y = np.array(Y)

# fit the model
model = Sequential()
model.add(LSTM(units=5, return_sequences=True, input_shape=(n_lookback, 1)))
model.add(LSTM(units=5))
model.add(Dense(n_forecast))
model.compile(loss='mean_squared_error', optimizer='adam') #use mse for loss function and adam optimizer
model.fit(X, Y, epochs=100, batch_size=16, verbose=0)

# generate the forecasts
X_ = y[- n_lookback:]  # last available input sequence
X_ = X_.reshape(1, n_lookback, 1) #run for 100 epochs

Y_ = model.predict(X_).reshape(-1, 1)
Y_ = scaler.inverse_transform(Y_)

# organize the results in a data frame
df_past = df[['longitude']].reset_index()
df_past.rename(columns={'date' : 'Date', 'longitude': 'Actual'}, inplace=True)
df_past['Date'] = pd.to_datetime(df_past['Date'])
df_past['Forecast'] = np.nan
df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]
df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
# df_future['Forecast'] = Y_.flatten()
df_future['Forecast'] = Y_
df_future['Forecast'] = df_future['Forecast'].apply(lambda x: x*-1)
df_future['Actual'] = np.nan

resultsLongitude = df_past.append(df_future).set_index('Date')

# save the results
resultsLongitude.to_csv('static/dataset/longitude.csv')
