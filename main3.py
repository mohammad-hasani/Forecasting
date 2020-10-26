import talib
import pandas as pd


def main():
    # Index
    djia = pd.read_excel('./main3/DJIA.xls')
    djia = djia.dropna()
    djia = djia.iloc[:, 1]

    # RSI
    rsi = talib.RSI(djia, timeperiod=14)

    djia = djia[14:]
    rsi = rsi[14:]



    print(djia)
    print(rsi)


if __name__ == '__main__':
    main()