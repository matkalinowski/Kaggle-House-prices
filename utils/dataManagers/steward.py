from utils.dataManagers.Data import DataObject
from pathlib import Path
import pandas as pd


def _read_data(data_type):
    return pd.read_csv(Path('data/' + data_type + '.csv').absolute())


def formulate_data(data_type, num_type_handling, categoricals_handling):
    do = DataObject(_read_data(data_type))
    return do.handle_missing_values(num_type_handling, categoricals_handling)


class DataSteward(object):
    def __init__(self, categoricals_handling=None, num_type_handling=None):
        self.train_data = formulate_data('train', num_type_handling, categoricals_handling)
        self.test_data = formulate_data('test', num_type_handling, categoricals_handling)

        self.train_response = self.train_data.SalePrice
        del self.train_data['SalePrice']
