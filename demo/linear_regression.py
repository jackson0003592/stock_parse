import pandas_datareader.data as web
import pandas as pd
import numpy as np
import datetime
import statsmodels.api as sm
from statsmodels import regression
import matplotlib.pyplot as plt


def main():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    df_stockload = web.DataReader("600797.SS", "yahoo", datetime.datetime(2018, 10, 1), datetime.datetime(2019, 4, 1))
    df_stockload.fillna(method='bfill', inplace=True)

    # print(df_stockload.info)

    y_arr = df_stockload.Close.values
    x_arr = np.arange(0, len(y_arr))



    x_b_arr = sm.add_constant(x_arr)

    print(x_b_arr)

    model = regression.linear_model.OLS(y_arr, x_b_arr).fit()
    rad = model.params[1]
    intercept = model.params[0]

    reg_y_fit = x_arr * rad + intercept

    plt.plot(x_arr, y_arr)
    plt.plot(x_arr, reg_y_fit, 'r')
    plt.title("浙大网新" + " y = " + str(rad) + " * x +" + str(intercept))
    plt.legend(['close', 'linear'], loc='best')

    plt.show()


if __name__ == '__main__':
    main()
