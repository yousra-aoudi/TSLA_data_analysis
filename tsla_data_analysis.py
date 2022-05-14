# 1 - Problem definition

"""
In the supervised regression framewor used for this data analysis case,
the weekly return of TSLA stock is the predicted variable.
We will try to understand what affects TSLA stock price and incorporate as much
information into the model. In this case study, we will focus on correlated assets as features.
For upcoming study cases, we will use technical indicators and fundamental analysis.

For this case study, other than the historical data of TSLA, the independent variables
used are the following correlated assets:
 -> Stocks: NIO and GM
 -> Currency : USD/JPY and GBP/USD
 -> Indices : S&P 500, Dow Jones, and VIX

The dataset used for this case study is extracted from Yahoo Finance and the FRED
website.
"""

# 2 - Loading the data and Python packages

# 2.1. - Loading the Python packages
"""
Below we indicate the list of the libraries used for data loading, data analysis,
data preparation, model evaluation, and model tuning.
"""

# Function and modules for the supervised regression models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor

# Function and modules for data analysis and model evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression

# Function and modules for deep learning models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import LSTM
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# Function and modules for time series models
from statsmodels import tsa
#from tsa import arima as ARIMA
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.arima_model import ARIMA


# Function and modules for data preparation and visualization
    # pandas, pandas_datareader, numpy and matplotlib
import numpy as np
import pandas as pd
#import pandas_datareader as web
import pandas_datareader.data as web
import pandas_datareader as pdr
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
import seaborn as sns
import scipy
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

# Yahoo for dataReader
import yfinance as yf
yf.pdr_override()

# Datetime
from datetime import *

#Diable the warnings
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# 2.2. - Loading the data
stock_tickers = ['TSLA','F','GM']
currency_tickers = ['DEXJPUS', 'DEXUSUK']
idx_tickers = ['SP500', 'DJIA','VIXCLS']

start_date = datetime(2018, 1, 1)
end_date = datetime(2022, 5, 1)

# Stock
stock_data = yf.download(stock_tickers,start=start_date,end=end_date)
df_stock_data = pd.DataFrame(data=stock_data).dropna()
print('Stock data describe \n',df_stock_data.describe())
# Outliers - cleaning
no_outlier_stock_data = df_stock_data[(np.abs(scipy.stats.zscore(df_stock_data)) < 2).all(axis=1)]
print('Outliers Stock data describe \n',no_outlier_stock_data.describe())
df_stock_data = no_outlier_stock_data
print('Stock data after removing outliers \n',df_stock_data.head())

# Currency
currency_data = web.get_data_fred(currency_tickers,start=start_date,end=end_date)
df_currency_data = pd.DataFrame(data=currency_data).dropna()
print('FX data describe \n',df_currency_data.describe())
# Outliers - cleaning
no_outlier_currency_data = df_currency_data[(np.abs(scipy.stats.zscore(df_currency_data)) < 2).all(axis=1)]
print('Outliers FX data describe \n',no_outlier_currency_data.describe())
df_currency_data = no_outlier_currency_data
print('Currency data after removing outliers \n',df_currency_data.head())

# Indices
idx_data = web.get_data_fred(idx_tickers,start=start_date,end=end_date)
df_idx_data = pd.DataFrame(data=idx_data).dropna()
print('Indices data describe \n',df_idx_data.describe())
# Outliers - cleaning
no_outlier_idx_data = df_idx_data[(np.abs(scipy.stats.zscore(df_idx_data)) < 2).all(axis=1)]
print('Outliers - Indices data describe \n',no_outlier_idx_data.describe())
df_idx_data = no_outlier_idx_data
print('Idx data after removing outliers \n',df_idx_data.head())

return_period = 5
Y = np.log(df_stock_data.loc[:,('Close','TSLA')]).diff(return_period).shift(-return_period)
Y.name = Y.name[-1]+'_pred'

X1 = np.log(df_stock_data.loc[:, ('Close', ('F', 'GM'))]).diff(return_period).dropna()
X1.columns = X1.columns.droplevel()
X2 = np.log(currency_data).diff(return_period).dropna()
X3 = np.log(idx_data).diff(return_period).dropna()

X4 = pd.concat([np.log(df_stock_data.loc[:, ('Close', 'TSLA')]).diff(i) for i in [return_period, return_period*3,
                                                                            return_period*6,return_period*12]],
               axis=1).dropna()
X4.columns = ['TSLA_DT', 'TSLA_3DT', 'TSLA_6DT', 'TSLA_12DT']
X = pd.concat([X1, X2, X3, X4], axis=1)
print('X columns \n',X.columns)

dataset = pd.concat([Y,X], axis=1).dropna().iloc[::return_period,:]
Y = dataset.loc[:,Y.name]
X = dataset.loc[:, X.columns]

# 3. Exploratory data analysis
"""
In this section, we will look at descriptive statistics, data visualization,
and time series analysis.
"""

# 3.1. - Descriptive statistics
print('dataset \n',dataset.head())
print('Statistics \n',dataset.describe())

# 3.2. Data visualization
"""
In order to learn more about our data, we will visualize it. 
Visualization involves independently understanding each attribute of
the dataset. Therefore, we will look at the scatterplot and the correlation
matrix. These plots will give us a sense of the interdependence of
the data. Correlation can be can be calculated and displayed for each pair of the variables
by creating a correlation matrix.
Besides the relationship between dependent and independent variables, it also
shows the correlation among the independent variables. 
"""
dataset.hist(bins=50, sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(12,12))
pyplot.savefig('TSLA - Data Visualization - Histograms.png')
pyplot.show()

dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=True, legend=True, fontsize=1, figsize=(15,15))
pyplot.savefig('TSLA - Data Visualization - Density.png')
pyplot.show()

# Correlation - To get a sense of data interdependence
correlation = dataset.corr()
pyplot.figure(figsize=(15,15))
pyplot.title('Correlation Matrix')
sns.heatmap(correlation,vmax=1,square=True,annot=True,cmap='cubehelix')
pyplot.savefig('TSLA - Correlation Matrix.png')
pyplot.show()

# 3.2. Data visualization
"""
In order to learn more about our data, we will visualize it. 
Visualization involves independently understanding each attribute of
the dataset. Therefore, we will look at the scatterplot and the correlation
matrix. These plots will give us a sense of the interdependence of
the data. Correlation can be can be calculated and displayed for each pair of the variables
by creating a correlation matrix.
Besides the relationship between dependent and independent variables, it also
shows the correlation among the independent variables. 
"""

correlation = dataset.corr()
pyplot.figure(figsize=(15,15))
pyplot.title('Correlation Matrix')
sns.heatmap(correlation,vmax=1,square=True,annot=True,cmap='cubehelix')
plt.show()

"""
Let us now visualize the relationship between all the variables in the
regression using the scatterplot matrix below.
"""


pyplot.figure(figsize=(15,15))
scatter_matrix(dataset,figsize=(12,12))
pyplot.savefig('TSLA - scatter-plot matrix.png')
plt.show()


# 3.3. Time series analysis

"""
Decomposition of the time series of the predicted variable into trend
and seasonality components.
"""

res = sm.tsa.seasonal_decompose(Y,freq=52)
fig = res.plot()
fig.set_figheight(8)
fig.set_figwidth(15)
pyplot.savefig('TSLA - Trend and seasonality components.png')
pyplot.show()

# Data preparation
"""
This step of data analysis typically involves data preprocessing, data cleaning,
looking at feature importance, and performing feature reduction.
"""

# 5. Models evaluation
# 5.1. Train-test split and evaluation metrics
"""
In this step, we will partition the original dataset into a training
set and a test set.
The test set is a sample of the data that we hold back from our analysis and
modeling.
We use it right at the end of our project to confirm the performance
of the final model.
It is the final test that gives us confidence on our estimates of
accuracy on useen data. 
We will use 80% of the dataset for modeling and use 20% for testing.
"""

validation_size = 0.2
train_size = int(len(X) * (1-validation_size))
X_train, X_test = X[0:train_size], X[train_size:len(X)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(X)]

# 5.2. Test options and evaluation metrics
"""
To optimize the various hyperparameters of the models, we use ten-fold
cross validation (CV) and recalculate the results ten times to account
for the inherent randomness in some of the models and the CV process.

The mean squared error metric will be used to evaluate the algorithms.
the MSE gives an idea of the performance of the supervised regression
models. 
"""

num_folds = 10
scoring = 'neg_mean_squared_error'

# 5.3. Compare models and algorithms
"""
After completion of data loading and designed the test harness, we will need
to choose a model. 
"""

# 5.3.1 Machine learning models from scikit-learn

# Regressio and tree regression algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO',Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

# Neural network algorithms
models.append(('MLP', MLPRegressor()))

# Ensemble models

# Boosting methods
models.append(('ABR', AdaBoostRegressor()))
models.append(('GBR', GradientBoostingRegressor()))

# Bagging methods
models.append(('RFR', RandomForestRegressor()))
models.append(('ETR', ExtraTreesRegressor()))

"""
Now that we slected all the models, we will loop over each of them. 
First, we run the k-fold analysis. 
Next, we run the model on the entire training and testing dataset.
We willl calculate the mean and standadr deviation of the evaluation metric
for each algorithm and collect the results for model comparison later:
"""

names = []
kfold_results = []
test_results = []
train_results = []
seed = 7
scoring = 'neg_mean_squared_error'
for name, model in models:
    names.append(name)
    # k-fold analysis:
    kfold = KFold(n_splits=num_folds, random_state=None)
    # Converted mean squared error to positive. The lower the better
    cv_results = -1*cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    kfold_results.append(cv_results)
    # Full training period
    res = model.fit(X_train, Y_train)
    train_result = mean_squared_error(res.predict(X_train), Y_train)
    train_results.append(train_result)
    # Test results
    test_result = mean_squared_error(res.predict(X_test),Y_test)
    test_results.append(test_result)

    msg = "%s: %f (%f) %f %f" % (name, cv_results.mean(), cv_results.std(), train_result, test_result)
    print(msg)

# Comparison of algorithms by looking at the cross validation results

# Cross validation results

fig = pyplot.figure()
fig.suptitle('Algorithm Comparison: Kfold results')
ax = fig.add_subplot(111)
pyplot.boxplot(kfold_results)
ax.set_xticklabels(names)
fig.set_size_inches(15,8)
pyplot.savefig('TSLA - Algorithm Comparison: Kfold results.png')
pyplot.show()

# Let us now looking at the errors of the test set as well.

# Training and test error

# compare algorithms
fig = pyplot.figure()

ind = np.arange(len(names)) # the x locations for the groups
width = 0.35 # the width of the bars

fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.bar(ind - width/2, train_results, width=width, label='Train Error')
pyplot.bar(ind + width/2, test_results, width=width, label='Test Error')
fig.set_size_inches(15,8)
pyplot.legend()
ax.set_xticks(ind)
ax.set_xticklabels(names)
pyplot.show()

# Time series–based models: ARIMA and LSTM
X_train_ARIMA = X_train.loc[:,['F','GM', 'DEXJPUS', 'SP500', 'DJIA','VIXCLS']]
X_test_ARIMA = X_test.loc[:,['F','GM', 'DEXJPUS', 'SP500', 'DJIA','VIXCLS']]
tr_len = len(X_train_ARIMA)
te_len = len(X_test_ARIMA)
to_len = len(X)

modelARIMA=ARIMA(endog=Y_train,exog=X_train_ARIMA,order=[1,0,0])
model_fit = modelARIMA.fit()

# Fitting the model
error_Training_ARIMA = mean_squared_error(Y_train, model_fit.fittedvalues)
predicted = model_fit.predict(start = tr_len -1 ,end = to_len -1, exog = X_test_ARIMA)[1:]
error_Test_ARIMA = mean_squared_error(Y_test,predicted)
print('error test \n',error_Test_ARIMA)

"""
seq_len = 2 #Length of the seq for the LSTM
Y_train_LSTM, Y_test_LSTM = np.array(Y_train)[seq_len-1:], np.array(Y_test)
X_train_LSTM = np.zeros((X_train.shape[0]+1-seq_len, seq_len, X_train.shape[1]))
X_test_LSTM = np.zeros((X_test.shape[0], seq_len, X.shape[1]))
for i in range(seq_len):
    X_train_LSTM[:, i, :] = np.array(X_train)[i:X_train.shape[0]+i+1-seq_len, :]
    X_test_LSTM[:, i, :] = np.array(X)[X_train.shape[0]+i-1:X.shape[0]+i+1-seq_len, :]

# LSTM Network
def create_LSTMmodel(learn_rate = 0.01, momentum=0): # create model
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train_LSTM.shape[1],X_train_LSTM.shape[2])))
    #More cells can be added if needed model.add(Dense(1))
    optimizer = SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss='mse', optimizer='adam')
    return model

LSTMModel = create_LSTMmodel(learn_rate = 0.01, momentum=0)
LSTMModel_fit = LSTMModel.fit(X_train_LSTM, Y_train_LSTM, validation_data=(X_test_LSTM, Y_test_LSTM),
                              epochs=330, batch_size=72, verbose=0, shuffle=False)

pyplot.plot(LSTMModel_fit.history['loss'], label='train', )
pyplot.plot(LSTMModel_fit.history['val_loss'], '--',label='test',)
pyplot.legend()
pyplot.savefig('TSLA - LSTM Model.png')
pyplot.show()

error_Training_LSTM = mean_squared_error(Y_train_LSTM,LSTMModel.predict(X_train_LSTM))
predicted = LSTMModel.predict(X_test_LSTM)
error_Test_LSTM = mean_squared_error(Y_test, predicted)
"""
test_results.append(error_Test_ARIMA)
#test_results.append(error_Test_LSTM)

train_results.append(error_Training_ARIMA)
#train_results.append(error_Training_LSTM)

names.append("ARIMA")

# compare algorithms
fig = pyplot.figure()

ind = np.arange(len(names))  # the x locations for the groups
width = 0.35  # the width of the bars

fig.suptitle('Comparing the performance of various algorthims on the Train and Test Dataset')
ax = fig.add_subplot(111)
pyplot.bar(ind - width/2, train_results,  width=width, label='Train Error')
pyplot.bar(ind + width/2, test_results, width=width, label='Test Error')
fig.set_size_inches(15,8)
pyplot.legend()
ax.set_xticks(ind)
ax.set_xticklabels(names)
pyplot.ylabel('Mean Square Error')
pyplot.savefig('Comparing the performance of various algorthims on the Train and Test Dataset.png')
pyplot.show()
#names.append("LSTM")


# Model Tuning and Grid Search

# Save Model for Later Use
# Save Model Using Pickle
from pickle import dump
from pickle import load

# save the model to disk
#filename = 'finalized_model.sav'
#dump(model_fit_tuned, open(filename, 'wb'))
"""
#Use the following code to produce the comparison of actual vs. predicted
predicted_tuned.index = Y_test.index
pyplot.plot(np.exp(Y_test).cumprod(), 'r') # plotting t, a separately
pyplot.plot(np.exp(predicted_tuned).cumprod(), 'b')
pyplot.rcParams["figure.figsize"] = (8,5)
pyplot.show()
"""

# estimate accuracy on validation set
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# prepare model
model_2 = Lasso()
model_2.fit(X_train, Y_train)
predictions_2 = model_2.predict(X_test)

mse_Lasso = mean_squared_error(Y_test, predictions_2)
r2_Lasso = r2_score(Y_test, predictions_2)
print("MSE Regression = %f " % (mse_Lasso))
print("R2 Regression = %f " % (r2_Lasso))

#Predictions - Lasso
train_size = int(len(X) * (1-validation_size))
X_train, X_test = X[0:train_size], X[train_size:len(X)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(X)]

model_lasso = Lasso()
model_lasso = model_lasso.fit(X_train, Y_train)
predictions = model.predict(X_test)

#Create column for Strategy Returns by multiplying the daily returns by the po
#of business the previous day
backtestdata = pd.DataFrame(index=X_test.index)
print('X_test \n', X_test.tail())
print('Y_test \n', Y_test.tail())
#backtestdata = pd.DataFrame()
backtestdata['close_pred'] = predictions
backtestdata['close_actual'] = Y_test
backtestdata['Market Returns'] = X_test['TSLA_DT'].pct_change()
backtestdata['Actual Returns'] = backtestdata['Market Returns'] * backtestdata
backtestdata['Strategy Returns'] = backtestdata['Market Returns'] * backtestda
#backtestdata=backtestdata.reset_index()
backtestdata.head()
backtestdata[['Strategy Returns','Actual Returns']].cumsum().hist()
backtestdata[['Strategy Returns','Actual Returns']].cumsum().plot()