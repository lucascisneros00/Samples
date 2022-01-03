#=================== PREDICTING S&P500 (SPY) ====================

#==== 0. DEPENDENCIES 
from datetime import datetime
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, os
from numpy.core.fromnumeric import shape, size
from tensorflow.python.keras.metrics import accuracy
import yfinance as yf

import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_squared_error as mse
from tensorflow import keras
from tensorflow.keras.callbacks import History 
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU

from tensorflow.keras import initializers
from numpy.random import seed

import warnings
import pmdarima as pm
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts

seed(123)
sns.set_theme(palette="viridis")
warnings.filterwarnings('ignore') 

#==== 1. DATA AND TESTS
# IMPORTING
data = yf.download('SPY','2021-01-01','2021-12-27')  #Data with SPY daily prices in 2021
prices = pd.DataFrame(data.Close.astype('float32'))

#prices = pd.DataFrame(np.arange(0,500,1)+np.random.uniform(0,1,500))   #Fake data for testing

# TESTS 
# Dickey-Fuller Test (DFT)
def dftest(timeseries):
    dftest = ts.adfuller(timeseries,)
    dfoutput = pd.Series(dftest[0:4], 
                         index=['Test Statistic','p-value','Lags Used','Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=30).mean()
    rolstd = timeseries.rolling(window=30).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries,label='Original')
    mean = plt.plot(rolmean, label='Rolling Mean')
    std = plt.plot(rolstd,  label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.grid(True)
    plt.show(block=False)

dftest(prices) #pval > 0.05: Cannot reject presence of unit root

# Autocorrelation and Partial Autocorrelation Plots
def plots(data, lags=None):
    layout = (1, 3)
    raw  = plt.subplot2grid(layout, (0, 0))
    acf  = plt.subplot2grid(layout, (0, 1))     
    pacf = plt.subplot2grid(layout, (0, 2))
    
    raw.plot(data)
    sm.tsa.graphics.plot_acf(data, lags=lags, ax=acf, zero=False)
    sm.tsa.graphics.plot_pacf(data, lags=lags, ax=pacf, zero = False)
    sns.despine()
    plt.tight_layout()
    plt.show(block=False)

plt.rcParams['figure.figsize'] = [11, 4]
plots(prices, lags=30)      #Very likely 1-day autocorr.
plt.rcParams['figure.figsize'] = plt.rcParamsDefault["figure.figsize"]



#==== 2. SPLITTING INTO TRAIN/TEST CHUNKS
FORECAST_LENGTH = 5    # y
SEQ_LENGTH = 30    # x + y
NUM_SEQ = 1000  # (x.shape = (100,29), y.shape= (100,1))

def get_sequences(data, num_seq=NUM_SEQ, seq_length=SEQ_LENGTH, forecast_length=FORECAST_LENGTH):
    idxs = np.random.randint(0,prices.shape[0]-seq_length+1, size=num_seq)  #Initializing random start of seq
    df = np.zeros(shape=(num_seq,seq_length))
    for i in range(num_seq):
        df[i,:]= np.array(prices.iloc[idxs[i]:idxs[i]+seq_length]).flatten()
    X = df[:,:-forecast_length]
    y = df[:, -forecast_length]
    return X, y

X, y  = get_sequences(prices)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1)

# Reshaping to Keras format
def get_keras_format_series(series):
    """
    Convert a series to a numpy array of shape 
    [n_samples, time_steps, features]
    """
    
    series = np.array(series)
    return series.reshape(series.shape[0], series.shape[1], 1)

#==== 3. TRAINING 
def measure_error(y_true, y_pred, label):
    return pd.Series({'MSE': mean_squared_error(y_true, y_pred),
                      'r2': r2_score(y_true, y_pred)},
                      name=label)

@tf.autograph.experimental.do_not_convert       #To silence a warning
def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )    #To prevent dividing by zero

# 1. RNN
# Architecture: 2 RNN Layers x25, 1 Dense Layer x5, (1 Final Dense Layer x1)
RNN_UNITS = 25
DENSE_UNITS = 5
BATCH_SIZE = 32
model_rnn = Sequential()
model_rnn.add(SimpleRNN(RNN_UNITS,
                    kernel_initializer=initializers.RandomNormal(stddev=0.001),
                    recurrent_initializer=initializers.Identity(gain=1.0),
                    activation='relu',
                    input_shape=get_keras_format_series(X_train).shape[1:]))
model_rnn.add(SimpleRNN(RNN_UNITS))
model_rnn.add(Dense(DENSE_UNITS))
model_rnn.add(Dense(1))
model_rnn.summary()

    # Optimizer
adam = keras.optimizers.Adam(learning_rate=0.001)
rmsprop = keras.optimizers.RMSprop(learning_rate = 0.0001)

# Compiling
model_rnn.compile(loss='mean_squared_error',
              optimizer=rmsprop,
              metrics=[r_squared])

# Fitting

rnn_history = History()
model_rnn.fit(get_keras_format_series(X_train), y_train,
          batch_size=BATCH_SIZE,
          epochs=100,
          validation_data=(get_keras_format_series(X_test), y_test),
          callbacks = [rnn_history])

# TESTING
y_pred = model_rnn.predict(get_keras_format_series(X_test)).flatten()
measure_error(y_test, y_pred, 'RNN')

sns.scatterplot( x=y_test, y=y_pred,)
ax = plt.gca()
ax.set(xlabel='y_true', ylabel='y_pred', title=f'R_Squared: {r2_score(y_test, y_pred):.4f}')
plt.show()
pd.DataFrame([y_pred, y_test])


# 2. LSTM
# Architecture: 2 LSTM Layers x25, 1 Dense Layer x5, (1 Final Dense Layer x1)
LSTM_UNITS = 25
DENSE_UNITS = 5
BATCH_SIZE = 32
model_lstm = Sequential()
model_lstm.add(LSTM(LSTM_UNITS,
                    kernel_initializer=initializers.RandomNormal(stddev=0.001),
                    recurrent_initializer=initializers.Identity(gain=1.0),
                    activation='relu',
                    input_shape=get_keras_format_series(X_train).shape[1:]))
model_lstm.add(LSTM(LSTM_UNITS))
model_lstm.add(Dense(DENSE_UNITS))
model_lstm.add(Dense(1))
model_lstm.summary()

    # Optimizer
adam = keras.optimizers.Adam(learning_rate=0.001)
rmsprop = keras.optimizers.RMSprop(learning_rate = 0.0001)

# Compiling
model_lstm.compile(loss='mean_squared_error',
              optimizer=adam,
              metrics=[r_squared])

# Fitting
lstm_history = History()
model_lstm.fit(get_keras_format_series(X_train), y_train,
          batch_size=BATCH_SIZE,
          epochs=100,
          validation_data=(get_keras_format_series(X_test), y_test),
          callbacks = [lstm_history])

# TESTING

# Statistics
y_pred = model_lstm.predict(get_keras_format_series(X_test)).flatten()
measure_error(y_test, y_pred, 'LSTM')

# Scatter
sns.scatterplot( x=y_test, y=y_pred,)
ax = plt.gca()
ax.set(xlabel='y_true', ylabel='y_pred', title=f'R_Squared: {r2_score(y_test, y_pred):.4f}')
plt.show()

# Loss/R-squared path across epochs
lstm_history.history.keys()
r2 = lstm_history.history['val_r_squared']
loss = lstm_history.history['val_loss']


# TRAIN/TEST SPLITS
FORECAST = 5
START = 1600

train = prices[:-FORECAST]
test = prices[-FORECAST:]

from statsmodels.tsa.api import SimpleExpSmoothing

single = SimpleExpSmoothing(train).fit(optimized=True)
single_preds = single.forecast(len(test))
single_mse = mse(test, single_preds)
print("Predictions: ", single_preds)
print("MSE: ", single_mse)

from statsmodels.tsa.api import Holt

double = Holt(train).fit(optimized=True)
double_preds = double.forecast(len(test))
double_mse = mse(test, double_preds)
print("Predictions: ", double_preds)
print("MSE: ", double_mse)

triple = ExponentialSmoothing(train,
                              trend="additive",
                              seasonal="additive",
                              seasonal_periods=(5)).fit(optimized=True)
triple_preds = triple.forecast(len(test))
triple_mse = mse(test, triple_preds)
print("Predictions: ", triple_preds)
print("MSE: ", triple_mse)


plt.plot(prices.index[:-FORECAST], train, 'b--', label="train")
plt.plot(prices.index[-FORECAST:], test, color='orange', linestyle="--", label="test")
plt.plot(prices.index[-FORECAST:], triple_preds, 'r--', label="predictions")
plt.legend(loc='upper left')
plt.title("Triple Exponential Smoothing")
plt.grid(alpha=0.3);
plt.show()


from statsmodels.tsa.statespace.sarimax import SARIMAX
sar = sm.tsa.statespace.SARIMAX(prices, 
                                order=(1,0,1), 
                                seasonal_order=(0,0,1,30), 
                                trend='t').fit()
sar_preds = sar.forecast(len(test))
sar_mse = mse(test, sar_preds)
#print("Predictions: ", sar_preds)
print("MSE: ", sar_mse)


#==== 2. TRAINING
# Function that returns metrics for validation
def measure_error(y_true, y_pred, label):
    return pd.Series({'roc_auc':roc_auc_score(y_true, y_pred),
                      'precision': precision_score(y_true, y_pred),
                      'recall': recall_score(y_true, y_pred),
                      'r2': r2_score(y_true, y_pred)},
                      name=label)
models = [None] * 4         #List of tuned models


#==== 3. RESULTS
# COMPUTING METRICS
y_hat = [None] * 4
aux = [None] * 4

for i in range(4):
    y_hat[i] = models[i].predict(X_test)
    aux[i] = measure_error(y_test, y_hat[i], 'Model '+str(i))

results = pd.concat([aux[0], aux[1], aux[2], aux[3]],
                              axis=1)
results.columns = ['Logistic Reg.', 'SVM', 'Rand. Forest', 'Voting Classifier']
print(results)

# PLOTTING
plot = results.unstack().reset_index() 
plot.columns = ["Model", "Metric", "Value"]
ax = sns.barplot(x="Model", y="Value", hue="Metric", data=plot)
ax.set(ylabel="")
plt.show()

r2s = list(results.iloc[3])
idx = r2s.index(max(r2s))
best_model = results.columns[idx]
print (f"\nBest model: {best_model}\nR-squared: {max(r2s):.4f}")