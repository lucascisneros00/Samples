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

#==== 1. DATA AND PREPROCESSING
# IMPORTING
data = yf.download('SPY','2012-01-01','2021-12-31')  #Data with SPY daily prices 2012-2021 (10 years)
prices = pd.DataFrame(data.Close.astype('float32'))
prices['day_of_week'] = prices.index.weekday       # Monday=0, Sunday=6

# Get sequences as weeks (starts at monday, ends on friday; sequences that include holidays are dropped (len(seq)<5))
'''
CODE HERE
'''
def get_weeks(prices):
    start = None
    end = None
    list = []
    week = prices['Close'].iloc[start:end]
    if len(week)==5: list.append(week)

    return list

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

dftest(prices); #pval > 0.05: Cannot reject presence of unit root

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
plot_autocorr(prices, lags=30)      #Very likely 1-day autocorr.
plt.rcParams['figure.figsize'] = plt.rcParamsDefault["figure.figsize"]



#==== 3. SPLITTING INTO TRAIN/TEST CHUNKS
# Predicts closing price given prices of the past month (5-day week)
SEQ_LENGTH = 20    #No. days as input (1 month)
NUM_SEQ = 1000  #No. simulations

def get_sequences(data, num_seq=NUM_SEQ, seq_length=SEQ_LENGTH):
    idxs = np.random.randint(0,prices.shape[0]-seq_length+1, size=num_seq)  #Initializing random start of sequence
    df = np.zeros(shape=(num_seq,seq_length))
    for i in range(num_seq):
        df[i,:]= np.array(prices.iloc[idxs[i]:idxs[i]+seq_length]).flatten()    #Simulate sequences
    X = df[:,:-1]
    y = df[:, -1]
    return X, y

X, y  = get_sequences(prices)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

# Reshaping X to Keras format (n_features=1)
def get_keras_format_series(series):
    """
    Convert a series to a numpy array of shape 
    [n_samples, time_steps, features]
    """
    
    series = np.array(series)
    return series.reshape(series.shape[0], series.shape[1], 1)

#==== 3. MODELS
# METRICS 
def measure_error(y_true, y_pred, label):
    return pd.Series({'MSE': mean_squared_error(y_true, y_pred),
                      'r2': r2_score(y_true, y_pred)},
                      name=label)

@tf.autograph.experimental.do_not_convert       #To silence a warning
def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )    #To prevent dividing by zero

# BUILDING
# 1. RNN
# Architecture: 1 RNN Layer x64, 1 Dense Layer x5, (1 Final Dense Layer x1)
RNN_UNITS = 64
DENSE_UNITS = 5
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
# Architecture: 1 LSTM Layer x64, 1 Dense Layer x5, (1 Final Dense Layer x1)
LSTM_UNITS = 64
DENSE_UNITS = 5
model_lstm = Sequential()
model_lstm.add(LSTM(LSTM_UNITS,
                    kernel_initializer=initializers.RandomNormal(stddev=0.001),
                    recurrent_initializer=initializers.Identity(gain=1.0),
                    activation='relu',
                    input_shape=get_keras_format_series(X_train).shape[1:]))
model_lstm.add(Dense(DENSE_UNITS))
model_lstm.add(Dense(1))
model_lstm.summary()

# TRAINING
BATCH_SIZE = 32
def compile_and_fit(model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, epochs=1000, batch_size=BATCH_SIZE, optimizer='Adam'):
        # Optimizer
    if optimizer=='Adam': opt = keras.optimizers.Adam(learning_rate=0.001)
    if optimizer=='RMSProp': opt = keras.optimizers.RMSprop(learning_rate = 0.0001)

    # Compiling
    model.compile(loss='mean_squared_error',
                optimizer=opt,
                metrics=[r_squared])

    # Fitting
    history = History()
    model.fit(get_keras_format_series(X_train), y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(get_keras_format_series(X_test), y_test),
            callbacks = [history])
    return model, history

model_rnn_fit, rnn_history = compile_and_fit(model=model_rnn, optimizer='RMSProp')
model_lstm_fit, lstm_history = compile_and_fit(model=model_lstm)

# ==== 4. RESULTS
# TESTING
y_hat_rnn = model_rnn.predict(get_keras_format_series(X_test)).flatten()
y_hat_lstm = model_lstm.predict(get_keras_format_series(X_test)).flatten()

print(measure_error(y_test, y_hat_rnn, 'RNN'))
print(measure_error(y_test, y_hat_lstm, 'LSTM'))

# FORECASTING NEXT WEEK'S PRICES
FORECAST_LENGTH = 10
def predict_t_steps(X_init, t_steps, model):
    """
    Given an input series matching the model's expected format,
    generates model's predictions for next t_steps in the series      
    """
    
    X_init = get_keras_format_series(X_init)
    preds = []
    
    # iteratively take current input sequence, generate next step pred,
    # and shift input sequence forward by a step (to end with latest pred).
    # collect preds as we go.
    for _ in range(t_steps):
        pred = model.predict(X_init)
        preds.append(pred)
        X_init[:,:-1,:] = X_init[:,1:,:]        #replace first t-1 values with 2nd through t-th
        X_init[:,-1,:] = pred       #replace t-th value with prediction
    
    preds = np.array(preds).reshape(-1,len(preds))
    
    return preds

y_pred_rnn = predict_t_steps(X_test, FORECAST_LENGTH, model_rnn)
y_pred_lstm = predict_t_steps(X_test, FORECAST_LENGTH, model_lstm)

start_range = range(1, X_test.shape[1]+1) #starting at one through to length of test_X_init to plot X_init
predict_range = range(X_test.shape[1],X_test.shape[1]+ FORECAST_LENGTH)  #predict range is going to be from end of X_init to length of test_hours

#using our ranges we plot X_init
sns.lineplot(start_range, X_test[0,:])

#and test and actual preds
#plt.plot(predict_range, y_test[0,:], color='orange')
sns.lineplot(predict_range, y_pred_rnn[0,:], linestyle='--', label="RNN forecast")      #NECESITO CAMBIAR LA FORMA EN QUE SEPARO TEST Y TRAIN.
sns.lineplot(predict_range, y_pred_lstm[0,:], linestyle='--', label="LSTM forecast")
plt.show()


# Scatter
sns.scatterplot( x=y_test, y=y_pred,)
ax = plt.gca()
ax.set(xlabel='y_true', ylabel='y_pred', title=f'R_Squared: {r2_score(y_test, y_pred):.4f}')
plt.show()
pd.DataFrame([y_pred, y_test])

# Loss/R-squared path across epochs
rnn_history.history.keys()
r2 = rnn_history.history['val_r_squared']
loss = rnn_history.history['val_loss']

r2_rolmean = pd.Series(r2).rolling(window=50).mean()   #50-epoch rolling mean
loss_rolmean = pd.Series(loss).rolling(window=50).mean()
sns.lineplot(y=r2, x=np.arange(len(r2)), linestyle='dotted', alpha=0.5, label='R2',)
sns.lineplot(y=r2_rolmean, x=np.arange(len(r2)), label='R2 (50-epoch rol. mean)')
sns.lineplot(y=loss, x=np.arange(len(loss)), linestyle='dotted', alpha=0.5, label='Loss',)
sns.lineplot(y=loss_rolmean, x=np.arange(len(loss)), label='Loss (50-epoch rol. mean)')
ax = plt.gca()
plt.ylim(bottom=0, top=1)
plt.xlim(left=50)
ax.set(xlabel='Epoch', title=f'RNN R-squared = {r2[-1]:.4f}')
plt.show()



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

r2_rolmean = pd.Series(r2).rolling(window=50).mean()   #50-epoch rolling mean
loss_rolmean = pd.Series(loss).rolling(window=50).mean()
sns.lineplot(y=r2, x=np.arange(len(r2)), linestyle='dotted', alpha=0.5, label='R2',)
sns.lineplot(y=r2_rolmean, x=np.arange(len(r2)), label='R2 (50-epoch rol. mean)')
sns.lineplot(y=loss, x=np.arange(len(loss)), linestyle='dotted', alpha=0.5, label='Loss',)
sns.lineplot(y=loss_rolmean, x=np.arange(len(loss)), label='Loss (50-epoch rol. mean)')
ax = plt.gca()
plt.ylim(bottom=0, top=1)
plt.xlim(left=50)
ax.set(xlabel='Epoch', title=f'LSTM R-squared = {r2[-1]:.4f}')
plt.show()


sns.lineplot(y=y_pred, x=np.arange(len(r2)), linestyle='dotted', alpha=0.5, label='R2',)
sns.lineplot(y=r2_rolmean, x=np.arange(len(r2)), label='R2 (50-epoch rol. mean)')

model_lstm.predict(get_keras_format_series(X_test)).shape


X_test[0].reshape()






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