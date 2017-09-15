from pathlib import Path

import pandas as pd

from utils.dataManagers.informations import informer


class DataSteward(object):
    def __init__(self):
        self.train_data = self._get_data('train')
        self.train_response = self.train_data.SalePrice
        del self.train_data['SalePrice']
        self.test_data = self._get_data('test')

    # def get_data_information(self):
    #     return informer.get_data_information_df()

    def _map_categorical_data(self, data):
        for col, mapper in informer.get_value_mappings().items():
            data[col].map(mapper)
            data[col] = data[col].astype('category')
        return data

    def _fill_null_vals_in_categorical_data(self, data):
        for col, mapper in informer.get_nan_value_mappings().items():
            data[col].fillna(mapper, inplace=True)
        return data

    def _get_data(self, data_type):
        data = pd.read_csv(Path('data/' + data_type + '.csv').absolute())
        data = self._fill_null_vals_in_categorical_data(data)
        data = self._map_categorical_data(data)

        return data


