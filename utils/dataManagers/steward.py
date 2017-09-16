from pathlib import Path

import pandas as pd

from utils.dataManagers.dataHelpers import NullsHandler


def _read_data(data_type):
    return pd.read_csv(Path('data/' + data_type + '.csv').absolute())


def set_id_as_index(df):
    df.index = df.Id
    del df['Id']
    return df


def formulate_data(data_type, num_type_handling, categoricals_handling):
    data = _read_data(data_type)
    data = NullsHandler(data, num_type_handling, categoricals_handling).handle_missing_values()
    return set_id_as_index(data)


class DataSteward(object):
    def __init__(self, categoricals_handling=None, num_type_handling=None):
        self.train_data = formulate_data('train', num_type_handling, categoricals_handling)
        self.test_data = formulate_data('test', num_type_handling, categoricals_handling)

        self.train_response = self.train_data.SalePrice
        del self.train_data['SalePrice']
