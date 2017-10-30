import numpy as np


def get_number_types(df):
    return df.select_dtypes(['int64', 'float64'])


def get_categoricals(df):
    return df.select_dtypes(['category'])


def log_given_columns(df, cols):
    ret_df = df.copy()
    ret_df[cols] = np.log1p(ret_df[cols])
    return ret_df


def restore_predictions_from_log1p(predictions):
    return np.expm1(predictions)
