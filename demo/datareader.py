import pandas_datareader.data as web
import datetime
import numpy as np

df_stockload = web.DataReader("000001.SS", "yahoo", datetime.datetime(2018, 1, 1), datetime.datetime(2019, 1, 1))
df_stockload['Ma20'] = df_stockload.Close.rolling(window=20).mean()
df_stockload.dropna(axis=0, how='any', inplace=True)

def forin_looping(df):
    df['signal'] = 0 #df = df.assign(signal = 0)  #可采用assign新增一列
    for i in np.arange(0,df.shape[0]):
        df.iloc[i,df.columns.get_loc('signal')] = np.sign(df.iloc[i]['Close'] - df.iloc[i]['Ma20'])
    return df

print(forin_looping(df_stockload)[0:5])

