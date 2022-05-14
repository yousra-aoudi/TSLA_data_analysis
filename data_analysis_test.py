# Checking data
# Read stocks data
input_path_stock_data = '/Users/yousraaoudi/Desktop/TSLA_Data_analysis.py/stock_data.csv'
df_temp_stock_data = pd.read_csv(input_path_stock_data, index_col='Date', parse_dates=True, na_values=['nan']).dropna()
print(" Missing value \n", df_stock_data.isnull().values.any())
print('df stock data \n', df_temp_stock_data)
df_stock_data = df_stock_data.join(df_temp_stock_data)
# drop dates ETF didn't trade
#df = df.dropna()

print('stock data description \n',df_stock_data.describe())
print('stock data description \n',df_stock_data.columns)

#stock_data = pd.read_csv(input_path_stock_data).dropna() #read the data from the supplied file
print(df_stock_data.head())

# Currency

df_currency_data = pd.DataFrame()

# Checking data
print(" Missing value \n", df_currency_data.isnull().values.any())
# Read currency index data

input_path_currency_data = '/Users/yousraaoudi/Desktop/TSLA_Data_analysis.py/currency_data.csv'
df_temp_currency_data = pd.read_csv(input_path_currency_data, index_col='Date',parse_dates=True,
                                    na_values=['nan']).dropna()
print('df stock data \n', df_temp_currency_data)
df_currency_data = df_currency_data.join(df_temp_currency_data)
# drop dates ETF didn't trade
print(df_currency_data.head())

# Idx

df_idx_data = pd.DataFrame()

# Checking data
print(" Missing value \n", df_idx_data.isnull().values.any())
# Read ETF data

input_path_idx_data = '/Users/yousraaoudi/Desktop/TSLA_Data_analysis.py/idx_data.csv'
df_temp_idx_data = pd.read_csv(input_path_idx_data, index_col='Date',parse_dates=True,na_values=['nan']).dropna()
print('df stock data \n', df_temp_idx_data)
df_idx_data = df_idx_data.join(df_temp_idx_data)
# drop dates ETF didn't trade
print(df_idx_data.head())

#stock_data = pdr.get_data_yahoo(stock_tickers, start=start_date, end=end_date)
stock_data = yf.download(stock_tickers,start=start_date,end=end_date)

stock_data.to_csv('/Users/yousraaoudi/Desktop/TSLA_Data_analysis.py/stock_data.csv')

"""
for stock in stock_tickers:
    stock_data.to_csv('/Users/yousraaoudi/Desktop/TSLA_Data_analysis.py/'+str({stock})+'.csv',stock)
"""


#stock_data = pdr.get_data_yahoo(stock_tickers)
currency_data = web.get_data_fred(currency_tickers,start=start_date,end=end_date)
currency_data.to_csv('/Users/yousraaoudi/Desktop/TSLA_Data_analysis.py/currency_data.csv')

idx_data = web.get_data_fred(idx_tickers,start=start_date,end=end_date)
idx_data.to_csv('/Users/yousraaoudi/Desktop/TSLA_Data_analysis.py/idx_data.csv')


df_stock_data = pd.DataFrame()