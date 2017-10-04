import pandas as pd
from sklearn.preprocessing import Imputer

from utils.dataManagers.informations import informer

CATEGORICAL_TYPE_HANDLING_OPTIONS = [None, 'mapper_only', 'mapper_and_most_frequent']
NUMBER_TYPE_HANDLING_OPTIONS = [None, 'mean', 'median', 'most_frequent']

nan_mappings = informer.get_nan_value_mappings()
values_mappings = informer.get_value_mappings()


class NullsHandler(object):
    def __init__(self, data, num_type_handling, categoricals_handling):
        self.check_null_handling_options(categoricals_handling, num_type_handling)

        self.categoricals_handling = categoricals_handling
        self.num_type_handling = num_type_handling
        self.data = data

    @staticmethod
    def check_null_handling_options(categoricals_handling, num_type_handling):
        if categoricals_handling not in CATEGORICAL_TYPE_HANDLING_OPTIONS:
            raise AttributeError(f'{categoricals_handling} is not acceptable null handling option.'
                                 f'\nUse one of: {CATEGORICAL_TYPE_HANDLING_OPTIONS}')
        if num_type_handling not in NUMBER_TYPE_HANDLING_OPTIONS:
            raise AttributeError(f'{num_type_handling} is not acceptable null handling option.'
                                 f'\nUse one of: {NUMBER_TYPE_HANDLING_OPTIONS}')

    @staticmethod
    def _map_categorical_data(col):
        return col.replace((values_mappings[col.name]))

    @staticmethod
    def _map_as_categories(col):
        return col.astype('category')

    @staticmethod
    def _fill_not_mapped_null_values(col) -> pd.Series:
        return col.fillna(col.value_counts().argmax())

    def handle_missing_values(self):

        categoricals = self.data[informer.get_value_mappings().index].copy()
        number_types = self.data.drop(list(informer.get_value_mappings().index), axis=1).copy()

        if self.categoricals_handling is 'mapper_only':
            categoricals = categoricals.fillna(nan_mappings) \
                .apply(self._map_categorical_data) \
                .apply(self._map_as_categories)
        elif self.categoricals_handling is 'mapper_and_most_frequent':
            categoricals = categoricals.fillna(nan_mappings) \
                .apply(self._map_categorical_data) \
                .apply(self._map_as_categories) \
                .apply(self._fill_not_mapped_null_values)

        if self.num_type_handling is not None:
            number_types = number_types.apply(self._fill_null_numerical_values, args=(self.num_type_handling,))

        return number_types.join(categoricals)

    @staticmethod
    def _fill_null_numerical_values(col, strategy):
        return Imputer(missing_values='NaN', strategy=strategy).fit_transform(col.values.reshape([-1, 1]))[:, 0]
