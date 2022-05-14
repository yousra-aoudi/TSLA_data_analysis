# 1 - Loading data - libraries

import os
import numpy as np
import pandas as pd
#import pandas_datareader as web
import pandas_datareader.data as web

#Diable the warnings
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# 2 - Loading the Data

"""
def symbol_to_path(symbol, base_dir="/Users/yousraaoudi/Desktop/TSLA_Data_analysis.py/"):
   Return CSV file path given data needed.
    symbol = ['stock_data','currency_data','idx_data']
    return os.path.join(base_dir, "{}".format(str(symbol)))

def load_financial_data(start_date, end_date,output_file): 
    try:
        df = pd.read_csv(output_file)
		print('File data found...reading data') 
	except FileNotFoundError:
		print('File not found...downloading the data')
		df = data.DataReader('GOOG', 'yahoo', start_date, end_date) 
		df.to_pickle(output_file)
	return df

goog_data=load_financial_data(start_date='2001-01-01', end_date = '2022-01-01',output_file='goog_data.pkl')

"""
def symbol_to_path(data, base_dir="/Users/yousraaoudi/Desktop/TSLA_Data_analysis.py/"):
    """Return CSV file path given data needed."""
    return os.path.join(base_dir, %f, %(str(data)))


def get_data(data):
    """Read ETF data (adjusted close) from CSV file."""

    df = pd.DataFrame()

    # Checking data
    print(" Missing value \n", df.isnull().values.any())
    # Read ETF data

    df_temp = pd.read_csv(symbol_to_path(data), index_col='Date', parse_dates=True, na_values=['nan'])
    print('df \n', df_temp)
    df = df.join(df_temp)
    # drop dates ETF didn't trade
    df = df.dropna()
    print("Negative adjusted values \n ",df[(df['Adj Close'] < 0)].sum)

    return df

# Read data
data = ['stock_data','currency_data','idx_data']
df = get_data(data)  # get data for each symbol
print('df \n', df.tail())