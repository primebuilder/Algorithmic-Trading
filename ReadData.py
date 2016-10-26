import os
import pandas as pd
import numpy as np
import warnings
import Regressors
from sklearn import model_selection
from sklearn import preprocessing
import matplotlib.pyplot as plt

warnings.simplefilter(action = "ignore", category = FutureWarning)

def fill_missing_values(df_data):
    """Fill missing values in data frame, in place."""
    df_data.fillna(method='ffill', inplace=True)
    df_data.fillna(method='bfill', inplace=True)


def get_data(symbols, dates):
    df = pd.DataFrame(index=dates)
    data = []
    for symbol in symbols:
        df_temp = pd.read_csv(os.path.join("Data/table_{}.csv".format(str(symbol))), index_col='Date',
                              parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_comp = pd.read_csv(os.path.join("Data/table_{}.csv".format(str(symbol))), index_col='Date',
                              parse_dates=True, na_values=['nan'])
        data.append(df_comp)
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
    return [df, data]


def smavg(df, N):
    return pd.rolling_mean(df, N)[N - 1:]


def expmavg(df, span):
    return pd.ewma(df, span=span)


def com_daily_returns(df):
    daily_returns = (df / df.shift(1)) - 1
    return daily_returns


def com_momentum(df, n):
    mom = (df / df.shift(n))
    return mom


def com_rsi(df, n):
    df = df.diff()
    df = df[1:]
    up, down = df.copy(), df.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = pd.rolling_mean(up, n)
    roll_down = pd.rolling_mean(down.abs(), n)
    rs = roll_up / roll_down
    rss = 100.0 - (100.0 / (1.0 + rs))
    return rss

def BBW(data,length):
    return 4*pd.stats.moments.rolling_std(data,length)

def com_williamspercent(df):
    high=pd.rolling_max(df,2)
    low=pd.rolling_min(df,2)
    return ((high-df)/(high-low))*100

def generate_data(df, symbols,compdata, n):
    data_list = [None]*len(symbols)
    for i in range(0, len(symbols)):
        temp_df = df.ix[:,[i]]
        comp_temp = compdata[i]
        volume = comp_temp.ix[:, ['Volume']]
        high = comp_temp.ix[:,['High']]
        close = comp_temp.ix[:,['Close']]
        low = comp_temp.ix[:,['Low']]
        # print type(volume)
        sum_data = temp_df.join(high)
        sum_data =sum_data.join(close)
        sum_data =sum_data.join(low)
        sum_data =sum_data.join(volume)
        fill_missing_values(sum_data)
        sum_data = sum_data[['High','Low','Close','Volume']]
        low = sum_data[['Low']]
        high =sum_data[['High']]
        volume = sum_data[['Volume']]
        close = sum_data[['Close']]
        sma10 = smavg(temp_df, 10)
        sma10 = sma10.rename(columns={symbols[i]: 'sma10'})
        sma20 = smavg(temp_df, 20)
        sma20 = sma20.rename(columns={symbols[i]: 'sma20'})
        sma30 = smavg(temp_df, 30)
        sma30 = sma30.rename(columns={symbols[i]: 'sma30'})
        ema10 = expmavg(temp_df, 10)
        ema10 = ema10.rename(columns={symbols[i]: 'ema10'})
        ema20 = expmavg(temp_df, 20)
        ema20 = ema20.rename(columns={symbols[i]: 'ema20'})
        ema30 = expmavg(temp_df, 30)
        ema30 = ema30.rename(columns={symbols[i]: 'ema30'})
        mom = com_momentum(temp_df, 10)
        mom = mom.rename(columns={symbols[i]: 'mom'})
        drr = com_daily_returns(temp_df)
        drr = drr.rename(columns={symbols[i]: 'drr'})
        bbw = BBW(temp_df, 20)
        bbw = bbw.rename(columns={symbols[i]: 'bbw'})
        will = com_williamspercent(temp_df)
        will = will.rename(columns={symbols[i]: 'will'})
        rsi = com_rsi(temp_df, 10)
        rsi = rsi.rename(columns={symbols[i]: 'rsi'})
        rsi = rsi.dropna()
        # print rsi.shape
        rsi = rsi.join(drr)
        rsi = rsi.join(mom)
        rsi = rsi.join(bbw)
        # rsi = rsi.join(will)
        rsi = rsi.join(ema10)
        rsi = rsi.join(ema20)
        rsi = rsi.join(ema30)
        rsi = rsi.join(sma10)
        rsi = rsi.join(sma20)
        rsi = rsi.join(sma30)
        # print rsi.shape
        rsi = rsi.join(volume)
        rsi = rsi.join(high)
        rsi = rsi.join(close)
        temp_df = temp_df.shift(-n)
        #print temp_df
        rsi = rsi.join(temp_df)
        # print rsi.shape
        rsi = rsi.dropna()
        data_list[i] = rsi
        # print rsi.shape
    return data_list

def plot_data(df):
    df.plot()
    plt.show()


def plotbollinger(df, symbols):
    for symbol in symbols:
        df1 = df[symbol]
        plt.clf()
        d = ' bollinger bands'
        e = symbol + d

        roll = pd.rolling_mean(df1, 50)
        rollstd = pd.rolling_std(df1, 50)
        upperB = roll + rollstd * 2
        lowerB = roll - rollstd * 2

        ax = df1.plot(title=e, label=symbol, color='blue')

        roll.plot(label="Rolling mean", color='black')
        upperB.plot(label="upper band", color='red')
        lowerB.plot(label="lower band", color='green')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc='upper left')
        # ax.legend(loc='upper left')
        a = './bollinger_bands/'
        b = symbol
        c = '.png'
        u = a + b + c
        print u
        directory = './bollinger_bands/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(u)

def test_run():
    symbols = ['A', 'AA', 'AAP', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AES', 'AET', 'AFL', 'AGN', 'AKAM', 'ALB', 'ALK', 'AMG',
               'APD', 'ATVI', 'AYI', 'LNT', 'MMM', 'S&P500']
    dates = pd.date_range('2011-10-13', '2016-10-12')
    [df, data] = get_data(symbols, dates)
    # print df['A']
    # print data
    fill_missing_values(df)
    # print df['ABBV']
    # plot_data(df['A'])
    plotbollinger(df,symbols)
    data_list = generate_data(df, symbols, data, 50)
    total_data = data_list[0]
    print data_list[1]
    for i in range(0,len(symbols)-1):
        temp_data = data_list[i]
        # print type(temp_data)
        test =  temp_data.ix[-50:,]
        train =  temp_data.ix[0:-50,]
        # print train.shape[1]
        train_data, train_label = train[train.columns[0:train.shape[1]-1]], train.ix[:, [train.shape[1]-1]]
        test_data, test_label = test[test.columns[0:test.shape[1]-1]], test.ix[:, [test.shape[1]-1]]
        # print type(test_label)
        # print test_data
        # Regressors.linear_regressor(train_data,train_label,test_data,test_label,symbols[i])
        # Regressors.poly_regressor(train_data,train_label,test_data,test_label,symbols[i])
        # Regressors.ridge_regressor(train_data, train_label, test_data, test_label, symbols[i])
        # Regressors.lasso_regressor(train_data, train_label, test_data, test_label,symbols[i])
        # Regressors.elasticnet_regressor(train_data, train_label, test_data, test_label, symbols[i])
        # Regressors.adaboost_regressor(train_data, train_label, test_data, test_label, symbols[i])
        # Regressors.svm_regressor(train_data, train_label, test_data, test_label,symbols[i])
        # Regressors.randomforest_regressor(train_data, train_label, test_data, test_label, symbols[i])
        # print test_label
        if(i>=1):
            total_data = total_data.join(data_list[i],rsuffix=symbols[i])

    print list(total_data.columns)
    sub='Volume'
    col_list = [s for s in list(total_data.columns) if sub in s]
    total_data[col_list] = total_data[col_list].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
    sp = data_list[len(symbols)-1]
    total_data = total_data.join(sp['S&P500'], rsuffix=symbols[len(symbols)-1])
    fill_missing_values(total_data)
    # print total_data
    print np.isnan(total_data.any()).tolist()
    print np.isfinite(total_data.all()).tolist()
    test = total_data.ix[-50:, ]
    train = total_data.ix[0:-50, ]
    print train.shape[1]
    train_data, train_label = train[train.columns[0:train.shape[1] - 1]], train.ix[:, [train.shape[1] - 1]]
    test_data, test_label = test[test.columns[0:test.shape[1] - 1]], test.ix[:, [test.shape[1] - 1]]
    # print Regressors.linear_regressor(train_data,train_label,test_data,test_label,symbols[len(symbols)-1])
    # print Regressors.poly_regressor(train_data,train_label,test_data,test_label,symbols[len(symbols)-1])
    # print Regressors.ridge_regressor(train_data, train_label, test_data, test_label, symbols[len(symbols)-1])
    # print Regressors.lasso_regressor(train_data, train_label, test_data, test_label,symbols[len(symbols)-1])
    # print Regressors.elasticnet_regressor(train_data, train_label, test_data, test_label, symbols[len(symbols)-1])
    # print Regressors.adaboost_regressor(train_data, train_label, test_data, test_label, symbols[len(symbols)-1])
    # print Regressors.svm_regressor(train_data, train_label, test_data, test_label,symbols[len(symbols)-1])
    # print Regressors.randomforest_regressor(train_data, train_label, test_data, test_label, symbols[len(symbols)-1])


if __name__ == "__main__":
    test_run()
