import pandas as pd

from utils.dataManagers.informations import mapper


def get_value_mappings():
    return pd.Series(mapper.value_mappings)


def get_nan_value_mappings():
    return pd.Series(mapper.nan_value_mappings)


def get_column_descriptions():
    return pd.Series(mapper.column_descriptions)


# def get_data_information_df():
#     col_desc = get_column_descriptions()
#     col_mappers = get_value_mappings()
#     nan_mappings = get_nan_value_mappings()
#
#     df = pd.DataFrame([col_desc, col_mappers, nan_mappings]).T
#
#     # #add column with data type of column
#     # for c in col_mappers.keys():
#     #     df.loc[c]['field_type'] = 'category'
#
#
#     df.columns = ['column_description', 'values_mapper', 'nan_mappings']
#     return df
