import pandas as pd
from data_prep import features_disc as features
from sklearn.preprocessing import MinMaxScaler


class data_preparation(object):

    def __init__(self, filepath: str, window_size: int, feature_window_size: int, discretize: bool):
        self.filepath = filepath
        self.window = window_size
        self.feature_window_size = feature_window_size
        self.discretize = discretize

    def get_data(self, filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath)

    def create_label(self, row):
        row['class'] = 1 if row['adjusted_close'] < row['shifted_value'] else -1
        return row

    def calculate_label_profit_loss(self, x):
        minimum = min(x)
        maximum = max(x)
        value = x[0]

        thresh_positive = value + value * 0.05
        thresh_negative = value - value * 0.05

        if maximum >= thresh_positive or minimum <= thresh_negative:
            for i in x:
                if i >= thresh_positive:
                    return 1
                elif i <= thresh_negative:
                    return -1
        else:
            return 0

    def create_label_profit_loss(self, df: pd.DataFrame, window):
        df['class'] = df['adjusted_close'].rolling(window=window).apply(lambda x: self.calculate_label_profit_loss(x))
        df['class'] = df['class'].shift(window - 1)

    def get_fresh_data_for_prediction(self, df: pd.DataFrame):
        result = df.where(df['shifted_value'].isna())
        result.dropna(thresh=1, inplace=True)
        return result

    def data_frame_with_features(self):
        df = self.get_data(self.filepath)

        df.drop(columns=["timestamp"], inplace=True)
        df.dropna(inplace=True)

        features.simpleMA(df, self.feature_window_size, self.discretize)
        features.weightedMA(df, self.feature_window_size, self.discretize)
        features.EMA(df, self.feature_window_size, self.discretize)
        features.momentum(df, self.feature_window_size)
        features.stochasticK(df, self.feature_window_size, self.discretize)
        features.stochasticD(df, self.feature_window_size, self.discretize)
        features.MACD(df, self.discretize)
        features.RSI(df, self.discretize)
        features.williamsR(df, 9, self.discretize)
        features.ADIndicator(df, self.discretize)
        features.diff_n_Months(df, 90)
        features.diff_current_lowest_low(df, 90)
        features.diff_current_highest_high(df, 90)
        features.standard_deviation(df, 90)
        features.skewness(df, 90)
        features.kurtosis(df, 90)
        features.entropy(df, 90)
        features.fourier_transform_min(df, 90)
        features.fourier_transform_max(df, 90)
        features.fourier_transform_mean(df, 90)
        features.CCI(df, 20, self.discretize)
        df.dropna(inplace=True)

        df['shifted_value'] = df['adjusted_close'].shift(-1 * self.window)
        df = scale_data(df)
        data_to_predict = self.get_fresh_data_for_prediction(df)
        self.create_label_profit_loss(df, self.window)
        df.dropna(inplace=True)
        df.drop(columns=['shifted_value', 'dividend_amount', 'split_coefficient', 'open', 'high', 'low', 'close',
                         '9-day-EMA', '12-day-EMA', '26-day-EMA', 'TP', 'TPMA', 'MeanDeviation'], inplace=True)
        data_to_predict.drop(
            columns=['shifted_value', 'dividend_amount', 'split_coefficient', 'open', 'high', 'low', 'close',
                     '9-day-EMA', '12-day-EMA', '26-day-EMA', 'TP', 'TPMA', 'MeanDeviation'], inplace=True)
        return df, data_to_predict


def scale_data(df: pd.DataFrame):
    df = df.copy()
    mms = MinMaxScaler()
    df[['diff_3_months', 'diff_LL', 'diff_HH', 'std', 'skew', 'kurtosis', 'entropy', 'fft_min', 'fft_max',
        'fft_mean']] = mms.fit_transform(df[['diff_3_months', 'diff_LL', 'diff_HH', 'std', 'skew', 'kurtosis',
                                             'entropy', 'fft_min', 'fft_max', 'fft_mean']])
    return df
