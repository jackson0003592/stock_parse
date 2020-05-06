import tushare as ts

df_tick = ts.get_tick_data('002372', date='2020-01-23')
print(df_tick)
