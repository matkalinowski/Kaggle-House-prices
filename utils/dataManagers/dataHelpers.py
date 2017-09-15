from utils.dataManagers.informations import informer
from sklearn.preprocessing import Imputer
import pandas as pd

CATEGORICAL_TYPE_HANDLING_OPTIONS = [None, 'mapper_only', 'mapper_and_most_frequent']
NUMBER_TYPE_HANDLING_OPTIONS = [None, 'mean', 'median', 'most_frequent']


def get_categoricals(df):
    return df.select_dtypes(['category'])


def get_nulls_df(input_df):
    return input_df.loc[input_df.isnull().any(axis=1), input_df.isnull().any()]


# Few things have to be corrected here:
# data must be parsed in memory,
# desirable format is:
# return data
#             .fill_null_values_in_categoricals()
#             .fill_null_values_in_numericals()
    
    
    
class NullsHandler(object):
    def __init__(self, data, num_type_handling, categoricals_handling):
        self.categoricals_handling = categoricals_handling
        self.num_type_handling = num_type_handling
        self.data = data

    @staticmethod
    def _map_categorical_data(data):
        for col, mapper in informer.get_value_mappings().items():
            data[col].map(mapper)
            data[col] = data[col].astype('category')
        return data

    @staticmethod
    def _fill_null_vals_in_categorical_data(data):
        for col, mapper in informer.get_nan_value_mappings().items():
            data[col].fillna(mapper, inplace=True)
        return data

    def _categoricals_filled_with_most_frequent_val(self, df):
        categoricals = get_categoricals(df).copy()
        for c in get_nulls_df(categoricals).columns:
            categoricals.loc[:, c] = df.loc[:, c].fillna(df.loc[:, c].value_counts().idxmax())
        return categoricals

    def handle_missing_values(self):
        categoricals = get_categoricals(self.data)
        if self.categoricals_handling is None:
            pass
        elif self.categoricals_handling is 'mapper_only':
            categoricals = self._map_categorical_data(self.data)
        elif self.categoricals_handling is 'mapper_and_most_frequent':
            categoricals = self._categoricals_filled_with_most_frequent_val(
                self._map_categorical_data(
                    self._fill_null_vals_in_categorical_data(self.data)))

        if self.num_type_handling is None:
            return self.data
        else:
            numerical = self.data.select_dtypes(['int64', 'float64'])
            return self._fill_null_numerical_values(numerical, self.num_type_handling)\
                .join(categoricals)

    def _fill_null_numerical_values(self, df, strategy):
        imr = Imputer(missing_values='NaN', strategy=strategy, axis=1)
        imr = imr.fit(df)
        return pd.DataFrame(imr.transform(df.values), columns=df.columns)
