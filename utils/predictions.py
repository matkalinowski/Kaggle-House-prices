import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from utils.dataManagers.dataWrangler import *


def get_df_for_predictions(train, test, standardize=True):
    all_data = pd.concat((train, test))

    if standardize:
        sd = StandardScaler()
        train_num_types = get_number_types(train)
        num_type_columns = train_num_types.columns
        train_num_types = sd.fit_transform(train_num_types)
        test_num_types = sd.transform(get_number_types(test))

        number_type = pd.DataFrame(np.concatenate((train_num_types, test_num_types)), columns=num_type_columns,
                                   index=all_data.index)
    else:
        number_type = get_number_types(all_data)

    categorical = get_categoricals(all_data)
    if categorical.shape[1] > 0:
        df = number_type.join(pd.get_dummies(categorical))
    else:
        df = number_type
    return df.iloc[:train.shape[0], :], df.iloc[train.shape[0]:, :]


def predict(results_class, clf, param_grid, xtrain, ytrain, xtest, name,
            plot_results=True, store_classifier=False, store_predictions=True,
            predictions_form_restoring_method=None, njobs=-1, cv=5):
    grid = GridSearchCV(clf, param_grid, cv=cv, scoring='neg_mean_squared_log_error', n_jobs=njobs)
    grid.fit(xtrain, ytrain)

    test_predictions = grid.predict(xtest)
    train_predictions = grid.predict(xtrain)
    if predictions_form_restoring_method is not None:
        test_predictions = predictions_form_restoring_method(test_predictions)
        train_predictions = predictions_form_restoring_method(train_predictions)
        ytrain = predictions_form_restoring_method(ytrain)

    results = results_class(grid, name, xtest.columns,
                            train_predictions, test_predictions, ytrain, store_classifier=store_classifier,
                            store_predictions=store_predictions)
    if plot_results:
        results.plot_results()
    return results
