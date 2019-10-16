import numpy as np
import pandas as pd
import scipy.stats as sc


def simpleMA(df: pd.DataFrame, moving_avg_window, discretize: bool):
    df['SMA'] = df['adjusted_close'].rolling(window=moving_avg_window).mean()
    # df.dropna(inplace=True)
    if discretize:
        df["SMA"] = (df['adjusted_close'] > df['SMA']).apply(lambda x: 1 if x else -1)


def weighted_calculations(x, moving_avg_window):
    wts = np.arange(start=1, stop=moving_avg_window + 1, step=1)
    return (wts * x).mean()


def weightedMA(df: pd.DataFrame, moving_avg_window, discretize: bool):
    df['WMA'] = df['adjusted_close'] \
        .rolling(window=moving_avg_window) \
        .apply(lambda x: weighted_calculations(x, moving_avg_window))
    # df.dropna(inplace=True)
    if discretize:
        df["WMA"] = (df['adjusted_close'] > df['WMA']).apply(lambda x: 1 if x else -1)


def EMA(df: pd.DataFrame, moving_avg_window, discretize: bool):
    df[f"{moving_avg_window}-day-EMA"] = df['adjusted_close'] \
        .ewm(span=moving_avg_window, adjust=False) \
        .mean()
    # df.dropna(inplace=True)
    if discretize:
        df[f"{moving_avg_window}-day-EMA"] = (df['adjusted_close'] > df[f"{moving_avg_window}-day-EMA"]).apply(
            lambda x: 1 if x else -1)


def momentum(df: pd.DataFrame, moving_window):
    df['Momentum'] = df['adjusted_close'].rolling(window=moving_window).apply(lambda x: x[0] - x[-1])
    df['Momentum'] = df['Momentum'].rolling(window=2).apply(lambda x: 1 if x[1] > x[0] else -1)
    # df.dropna(inplace=True)
    return df


def stochasticK_calculations(x):
    currentClose = x["adjusted_close"]
    highestHigh = x["highestHigh"]
    lowestLow = x["lowestLow"]
    high_low = (highestHigh - lowestLow)
    return (currentClose - lowestLow) / high_low


def stochasticK(df: pd.DataFrame, moving_window, discretize: bool):
    df["highestHigh"] = df["adjusted_close"].rolling(window=moving_window).apply(lambda x: max(x))
    df["lowestLow"] = df["adjusted_close"].rolling(window=moving_window).apply(lambda x: min(x))
    df.dropna(inplace=True)
    df["StochasticK"] = df[["adjusted_close", "highestHigh", "lowestLow"]] \
        .apply(lambda x: stochasticK_calculations(x), axis=1)
    df.drop(columns=["highestHigh", "lowestLow"], inplace=True)
    if discretize:
        df["StochasticK"] = df["StochasticK"].rolling(window=2).apply(lambda x: discretizeOscillator(x))
    # df.dropna(inplace=True)
    return df


def stochasticD(df: pd.DataFrame, moving_window, discretize: bool):
    if "StochasticK" in df:
        df["StochasticD"] = df["StochasticK"].rolling(3).mean()
        # df.dropna(inplace=True)
        if discretize:
            df["StochasticD"] = df["StochasticD"].rolling(window=2).apply(lambda x: discretizeOscillator(x))
        # df.dropna(inplace=True)
    else:
        return stochasticD(stochasticK(df, moving_window, discretize), moving_window, discretize)


def discretizeRSI(x):
    if x[-1] >= 70:
        return -1
    elif x[-1] <= 30:
        return 1
    elif x[-1] >= x[0]:
        return 1
    else:
        return -1


def RSI(df: pd.DataFrame, discretize: bool):
    df["GL"] = df["adjusted_close"].rolling(window=2).apply(lambda x: x[0] - x[1])
    df.dropna(inplace=True)
    df["RSI"] = df["GL"].rolling(window=14).apply(lambda x: calculateRSI(x))
    df.drop(columns=["GL"], inplace=True)
    # df.dropna(inplace=True)
    if discretize:
        df["RSI"] = df["RSI"].rolling(window=2).apply(lambda x: discretizeRSI(x))
        # df.dropna(inplace=True)
    return df


def calculateRSI(x):
    avgGain = abs(x[x > 0].sum() / 14)
    avgLoss = abs(x[x < 0].sum() / 14)
    RS = avgGain / avgLoss
    result = 100 - (100 / (1 + RS))
    return result


def discretizeMACD(x):
    return 1 if x[1] > x[0] else -1


def MACD(df: pd.DataFrame, discretize):
    EMA(df, 9, False)
    EMA(df, 12, False)
    EMA(df, 26, False)
    df.dropna(inplace=True)
    df['MACD'] = np.nan
    MACDLine = df['12-day-EMA'] - df['26-day-EMA']
    df['MACD'] = MACDLine - df['9-day-EMA']
    # df.dropna(inplace=True)

    if discretize:
        df["MACD"] = df["MACD"].rolling(window=2).apply(lambda x: discretizeMACD(x))

    # df.dropna(inplace=True)


def calculateWilliamsR(x):
    highestHigh = x["highestHigh"]
    lowestLow = x["lowestLow"]
    adjusted_close = x["adjusted_close"]
    return ((highestHigh - adjusted_close) / (highestHigh - lowestLow)) * -100


def discretizeOscillator(x):
    if x[-1] >= 70:
        return -1
    elif x[-1] <= 30:
        return 1
    elif x[-1] >= x[0]:
        return 1
    else:
        return -1


def discretizeWR(x):
    if x[-1] >= -20:
        return -1
    elif x[-1] <= -80:
        return 1
    else:
        if x[1] > x[0]:
            return 1
        else:
            return -1


def williamsR(df: pd.DataFrame, lookback_period: int, discretize: bool):
    df["highestHigh"] = df["adjusted_close"].rolling(window=lookback_period).max()
    df["lowestLow"] = df["adjusted_close"].rolling(window=lookback_period).min()
    # df.dropna(inplace=True)
    df["williamsR"] = df[["highestHigh", "lowestLow", "adjusted_close"]].apply(lambda x: calculateWilliamsR(x), axis=1)
    df.drop(columns=["highestHigh", "lowestLow"], inplace=True)
    if discretize:
        df["williamsR"] = df["williamsR"].rolling(window=2).apply(lambda x: discretizeWR(x))
        # df.dropna(inplace=True)


def ADIndicator(df: pd.DataFrame, discretize: bool):
    df["AD"] = ((df["close"] - df["low"]) - (df["high"] - df["close"]) / (df["high"] - df["low"]))
    df["AD"] = df["AD"].cumsum()
    if discretize:
        df["AD"] = df["AD"].rolling(window=2).apply(lambda x: 1 if x[0] < x[1] else -1)

def CCI(df: pd.DataFrame, window, discretize: bool):
    df["TP"] = (df['high'] + df['low'] + df['close'])/3
    df["TP"] = df["TP"].rolling(window=window).sum()
    df.dropna(inplace=True)

    df["TPMA"] = df["TP"].rolling(window=window).mean()
    df["MeanDeviation"] = df["TP"] - df["TPMA"]
    df["MeanDeviation"] = df["MeanDeviation"].rolling(window=window).mean()
    df.dropna(inplace=True)

    df["CCI"] = (df["TP"] - df["TPMA"]) / (0.15 * df["MeanDeviation"])
    if discretize:
        df["CCI"] = df["CCI"].rolling(window=2).apply(lambda x: discretize_CCI(x))

def discretize_CCI(x):
    if x[1] >= 200:
        return -1
    elif x[1] <= -200:
        return 1
    elif x[0] < x[1]:
        return 1
    else:
        return -1

def diff_n_Months(df: pd.DataFrame, window_size):
    df["diff_3_months"] = df["adjusted_close"].rolling(window=window_size).apply(lambda x: (x[-1] - x[0]) / x[-1])
    # df.dropna(inplace=True)


def diff_current_lowest_low(df: pd.DataFrame, window_size):
    df["diff_LL"] = df["adjusted_close"].rolling(window=window_size).apply(lambda x: (x[-1] - x.min()) / x[-1])
    # df.dropna(inplace=True)


def diff_current_highest_high(df: pd.DataFrame, window_size):
    df["diff_HH"] = df["adjusted_close"].rolling(window=window_size).apply(lambda x: (x[-1] - x.max()) / x[-1])
    # df.dropna(inplace=True)


def standard_deviation(df: pd.DataFrame, window_size):
    df["std"] = df["adjusted_close"] \
        .rolling(window=window_size) \
        .apply(lambda x: np.std(x))
    # df.dropna(inplace=True)


def skewness(df: pd.DataFrame, window_size):
    df["skew"] = df["adjusted_close"] \
        .rolling(window=window_size) \
        .apply(lambda x: sc.skew(x))
    # df.dropna(inplace=True)


def kurtosis(df: pd.DataFrame, window_size: int):
    df["kurtosis"] = df["adjusted_close"] \
        .rolling(window=window_size) \
        .apply(lambda x: sc.kurtosis(x))
    # df.dropna(inplace=True)


def entropy(df: pd.DataFrame, window_size: int):
    df["entropy"] = df["adjusted_close"] \
        .rolling(window=window_size) \
        .apply(lambda x: sc.entropy(x))
    # df.dropna(inplace=True)


def fourier_transform_min(df: pd.DataFrame, window_size):
    df["fft_min"] = df["adjusted_close"].rolling(window=window_size).apply(lambda x: np.min(np.absolute(np.fft.fft(x))))


def fourier_transform_max(df: pd.DataFrame, window_size):
    df["fft_max"] = df["adjusted_close"].rolling(window=window_size).apply(lambda x: np.max(np.absolute(np.fft.fft(x))))


def fourier_transform_mean(df: pd.DataFrame, window_size):
    df["fft_mean"] = df["adjusted_close"].rolling(window=window_size).apply(
        lambda x: np.mean(np.absolute(np.fft.fft(x))))


def checkValue(value):
    if value >= 0:
        return 1
    else:
        return -1
