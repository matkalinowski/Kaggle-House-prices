from utils.dataManagers.dataHelpers import NullsHandler


class DataObject(object):
    def __init__(self, data):
        self.data = data

    def handle_missing_values(self, num_type_handling, categoricals_handling):
        return NullsHandler(self.data, num_type_handling, categoricals_handling).handle_missing_values()
