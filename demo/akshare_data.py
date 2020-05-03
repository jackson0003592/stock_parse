import akshare as ak

# stock_df = ak.stock_zh_a_spot()
# print(stock_df)

stock_df = ak.stock_zh_a_daily(symbol='sh600000')
print(stock_df)