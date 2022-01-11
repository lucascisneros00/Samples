# Predicting the S&P500
I employ Deep Neural Networks, namely, RNN and LSTM architectures in order to predict next week's prices
of the SPY ETF (S&P 500 tracker), with only data of the previous month. In the future it would be interesting to
compare how these fare against traditional time series models (GARCH/ARIMA).

# Data
The data consists of daily prices of SPY for the last 20 years 2021 (5035 obs.), retrieved from Yahoo! Finance.
The models would severely improve with larger data sets, with minute-by-minute information for example.
