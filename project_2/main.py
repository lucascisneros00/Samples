#=================== PREDICTING S&P500 (SPY) ====================

#==== 0. DEPENDENCIES 
from datetime import datetime
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, os
import yfinance as yf

import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_squared_error as mse
from tensorflow import keras
from tensorflow.keras.callbacks import History, ModelCheckpoint
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

#==== 1. DATA AND PREPROCESSING
# IMPORTING
data = yf.download('SPY','2002-01-01','2021-12-31')  #Data with SPY daily prices 2002-2021 (20 years)
prices = pd.DataFrame(data.Close.astype('float32'))

#==== 2. TESTS
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

dftest(prices.Close); #pval > 0.05: Cannot reject presence of unit root

# Autocorrelation and Partial Autocorrelation Plots
def plot_autocorr(data, lags=None):
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
plot_autocorr(prices.Close, lags=30)      #Very likely 1-day autocorr.
plt.rcParams['figure.figsize'] = plt.rcParamsDefault["figure.figsize"]

#==== 3. SPLITTING INTO TRAIN/TEST CHUNKS
# Predicts closing price given prices of the past month (5-day week)
SEQ_LENGTH = 20    #No. days as input (1 month)
TEST_LENGTH = 20*12   #Use last year as test set
GAP_LENGTH = 5      #Some overlap between sequences
# Reshaping X to Keras format (n_features=1)
def get_keras_format_series(series):
    """
    Convert a series to a numpy array of shape 
    [n_samples, time_steps, features]
    """
    
    series = np.array(series)
    return series.reshape(series.shape[0], series.shape[1], 1)

def get_sequences(timeseries, seq_length=SEQ_LENGTH, test_length=TEST_LENGTH, gap=GAP_LENGTH):
    """
    Split a series into train and test in keras format
    """
    train = timeseries[:-test_length]   #training data is remaining months until amount of test_length
    test = timeseries[-test_length:]    #test data is the remaining test_length
    X_train, y_train = [], []

    for i in range(0, train.shape[0]-seq_length, gap):
        X_train.append(train[i:i+seq_length]) #each training sample is of length seq_length
        y_train.append(train[i+seq_length]) #each y is just the next step after training sample

    X_train = get_keras_format_series(X_train) # format our new training set to keras format
    y_train = np.array(y_train) # make sure y is an array to work properly with keras

    # Same process for test set
    X_test, y_test = [], []

    for i in range(0, test.shape[0]-seq_length, gap):
        X_test.append(test[i:i+seq_length]) #each training sample is of length seq_length
        y_test.append(test[i+seq_length]) #each y is just the next step after training sample

    X_test = get_keras_format_series(X_test) # format our new training set to keras format
    y_test = np.array(y_test) # make sure y is an array to work properly with keras

    return X_train, X_test, y_train, y_test, train, test

X_train, X_test, y_train, y_test, train, test  = get_sequences(prices.Close)

#==== 4. MODELS
# METRICS 
def measure_error(y_true, y_pred, label):
    return pd.Series({'MSE': mean_squared_error(y_true, y_pred),
                      'r2': r2_score(y_true, y_pred)},
                      name=label)

@tf.autograph.experimental.do_not_convert       #To silence a warning
def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )    #K.epsilon() To prevent dividing by zero

# BUILDING
# 1. RNN
# Architecture: 1 RNN Layer x64, 1 Dense Layer x5, (1 Final Dense Layer x1)
RNN_UNITS = 64
DENSE_UNITS = 10
model_rnn = Sequential()
model_rnn.add(SimpleRNN(RNN_UNITS,
                    kernel_initializer=initializers.RandomNormal(stddev=0.001),
                    recurrent_initializer=initializers.Identity(gain=1.0),
                    activation='relu',
                    input_shape=get_keras_format_series(X_train).shape[1:]))
model_rnn.add(Dense(DENSE_UNITS))
model_rnn.add(Dense(1))
model_rnn.summary()

# 2. LSTM
# Architecture: 1 LSTM Layer x64, 1 Dense Layer x10, (1 Final Dense Layer x1)
LSTM_UNITS = 64
DENSE_UNITS = 10
model_lstm = Sequential()
model_lstm.add(LSTM(LSTM_UNITS,
                    kernel_initializer=initializers.RandomNormal(stddev=0.001),
                    recurrent_initializer=initializers.Identity(gain=1.0),
                    activation='relu',
                    input_shape=get_keras_format_series(X_train).shape[1:]
                    ))                 
model_lstm.add(Dense(DENSE_UNITS))
model_lstm.add(Dense(1))
model_lstm.summary()

# TRAINING
BATCH_SIZE = 32
def compile_and_fit(model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, epochs=500, batch_size=BATCH_SIZE, optimizer='Adam'):
        # Optimizer
    if optimizer=='Adam': opt = keras.optimizers.Adam(learning_rate=0.001)
    if optimizer=='RMSProp': opt = keras.optimizers.RMSprop(learning_rate = 0.0001)

    # Compiling
    model.compile(loss='mean_squared_error',
                optimizer=opt,
                metrics=[r_squared])

    # Fitting
    '''
        # Callback to save the best model
    best_model_path = os.path.join('project_2','models', f'best_{model}')
    os.makedirs(best_model_path, exist_ok=True)
    best_model = ModelCheckpoint(
        filepath=best_model_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    '''
        
    history = History()
    model.fit(get_keras_format_series(X_train), y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(get_keras_format_series(X_test), y_test),
            callbacks = [history])
    #model.load_weights(best_model_path)
    return model, history

model_rnn_fit, rnn_history = compile_and_fit(model=model_rnn, optimizer='RMSProp')
model_lstm_fit, lstm_history = compile_and_fit(model=model_lstm)

# ==== 5. RESULTS
# TESTING
# Statistics
y_hat_rnn = model_rnn.predict(get_keras_format_series(X_test)).flatten()
y_hat_lstm = model_lstm.predict(get_keras_format_series(X_test)).flatten()

print(measure_error(y_test, y_hat_rnn, 'RNN'))
print(measure_error(y_test, y_hat_lstm, 'LSTM'))

# Scatters
sns.scatterplot( x=y_test, y=y_hat_rnn,)
ax = plt.gca()
x = np.linspace(*ax.get_xlim())
ax.plot(x, x)
ax.set(xlabel='y_true', ylabel='y_pred RNN', title=f'RNN R_Squared: {r2_score(y_test, y_hat_rnn):.4f}')
plt.show()

sns.scatterplot( x=y_test, y=y_hat_lstm,)
ax = plt.gca()
x = np.linspace(*ax.get_xlim())
ax.plot(x, x)
ax.set(xlabel='y_true', ylabel='y_pred LSTM', title=f'LSTM R_Squared: {r2_score(y_test, y_hat_lstm):.4f}')
plt.show()


# Loss/R-squared path across epochs
lstm_history.history.keys()
r2 = rnn_history.history['val_r_squared']
loss = rnn_history.history['val_loss']

r2_rolmean = pd.Series(r2).rolling(window=50).mean()   #50-epoch rolling mean
loss_rolmean = pd.Series(loss).rolling(window=50).mean()
sns.lineplot(y=r2, x=np.arange(len(r2)), linestyle='dotted', alpha=0.5, label='R2')
sns.lineplot(y=r2_rolmean, x=np.arange(len(r2)), label='R2 (50-epoch rol. mean)')
sns.lineplot(y=loss, x=np.arange(len(loss)), linestyle='dotted', alpha=0.5, label='Loss')
sns.lineplot(y=loss_rolmean, x=np.arange(len(loss)), label='Loss (50-epoch rol. mean)')
ax = plt.gca()
plt.ylim(bottom=0, top=1)
plt.xlim(left=50)
ax.set(xlabel='Epoch', title=f'LSTM Validation (Out-of-sample) R-squared = {r2[-1]:.4f}')
plt.show()


# FORECASTING NEXT WEEK'S PRICES
FORECAST_LENGTH = 5
def predict_t_steps(X_init, t_steps, model):
    """
    Given an input series matching the model's expected format,
    generates model's predictions for next t_steps in the series      
    """
    
    X = get_keras_format_series(X_init.copy())
    preds = []
    
    # iteratively take current input sequence, generate next step pred,
    # and shift input sequence forward by a step (to end with latest pred).
    # collect preds as we go.
    for _ in range(t_steps):
        pred = model.predict(X)
        preds.append(pred)
        X[:,:-1,:] = X[:,1:,:]        #replace first t-1 values with 2nd through t-th
        X[:,-1,:] = pred       #replace t-th value with prediction
    
    preds = np.array(preds).squeeze().T
    
    return preds

EXAMPLE_NO = 0 # We have X_test.shape[0]/GAP_LENGTH examples to choose from
def forecast_and_plot(X, forecast_length=FORECAST_LENGTH, n_example=EXAMPLE_NO):
    # Forecasts
    y_pred_rnn = predict_t_steps(X, forecast_length, model_rnn)[n_example,:]
    y_pred_lstm = predict_t_steps(X, forecast_length, model_lstm)[n_example,:]

    # True values
    y_true = test[(n_example+1)*(X.shape[1]+1):(n_example+1)*(X.shape[1]+1)+forecast_length]

    # Plot
    start_range = range(1, X.shape[1]+1) #starting at one through to length of X_test to plot X_test
    predict_range = range(X.shape[1],X.shape[1]+ forecast_length)  #predict range is going to be from end of X_init to length of test_hours
    #using our ranges we plot X_init
    sns.lineplot(start_range, X[n_example,:].flatten(), color='black')

    #and test and actual preds
    sns.lineplot(predict_range, y_true, color='blue', label='Actual value')
    sns.lineplot(predict_range, y_pred_rnn, color='red', linestyle='--', label="RNN forecast") 
    sns.lineplot(predict_range, y_pred_lstm, color='green', linestyle='--', label="LSTM forecast")
    ax = plt.gca()
    ax.set(xlabel='Time', ylabel='SPY Closing Price', title='Forecasting next week prices with data of the previous month')
    plt.show()
    return y_true, y_pred_rnn, y_pred_lstm
print(forecast_and_plot(X_test,FORECAST_LENGTH,EXAMPLE_NO))

# ... For the future: compare results with traditional ARIMA/GARCH models