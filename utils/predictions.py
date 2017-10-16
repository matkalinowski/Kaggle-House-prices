import datetime

import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from utils.dataManagers.dataWrangler import *


def predict_with_kfold(results_class, model, param_grid, X, y, test, folds=5, yield_progress=True,
                       plot_best_results=True, predictions_form_restoring_method=None, n_jobs=-1, stoppingRounds=None):
    start = datetime.datetime.now()
    kf = KFold(n_splits=folds, random_state=None, shuffle=False)
    errs = []
    res = []
    for train_index, test_index in kf.split(X):
        if yield_progress:
            print(f'iteration: {len(errs)} for class {type(model)}, time is: {datetime.datetime.now()}')
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        results = predict(results_class, model, param_grid, x_train,
                          y_train, test, stoppingRounds, name=None,
                          predictions_form_restoring_method=predictions_form_restoring_method,
                          plot_results=False, n_jobs=n_jobs)
        errs.append(mean_squared_log_error(results.grid.predict(x_test), y_test))
        res.append(results)
    errors = pd.Series(np.sqrt(errs))
    best_result_indx = errors.argmin()
    if yield_progress:
        print(f'Best test error for model: {type(model)} is: {errors.min()}')
        print('Refitting model.')
    best_result = res[best_result_indx]
    best_result.refit(X, y)
    end = datetime.datetime.now()
    if yield_progress:
        print(f'Calculations started at: {start} and ended at {end}, took: {end-start}')
    if plot_best_results:
        best_result.plot_results()
    return best_result


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


def predict(results_class, clf, param_grid, xtrain, ytrain, xtest, stoppingRounds=None, name=None,
            plot_results=True, store_classifier=False, store_predictions=True,
            predictions_form_restoring_method=None, n_jobs=-1, cv=5):
    if stoppingRounds:
        print('Performing RandomizedSearchCV.')
        grid = RandomizedSearchCV(clf, param_grid, cv=cv, scoring='neg_mean_squared_log_error', n_jobs=n_jobs,
                                  n_iter=stoppingRounds)
    else:
        print('Performing GridSearchCV.')
        grid = GridSearchCV(clf, param_grid, cv=cv, scoring='neg_mean_squared_log_error', n_jobs=n_jobs)
    grid.fit(xtrain, ytrain)

    test_predictions = pd.Series(grid.predict(xtest), index=xtest.index)
    train_predictions = pd.Series(grid.predict(xtrain), index=xtrain.index)

    restored_dict = {}
    if predictions_form_restoring_method is not None:
        restored_dict = save_restored_predictions_in_dict(predictions_form_restoring_method, test_predictions,
                                                          train_predictions, ytrain)

    results = results_class(grid, name, xtest.columns,
                            train_predictions, test_predictions, ytrain, restored_dict,
                            store_classifier=store_classifier,
                            store_predictions=store_predictions)
    if plot_results:
        results.plot_results()
    return results


def save_restored_predictions_in_dict(predictions_form_restoring_method, test_predictions, train_predictions, ytrain):
    return {'test_predictions': predictions_form_restoring_method(test_predictions),
            'train_predictions': predictions_form_restoring_method(train_predictions),
            'ytrain': predictions_form_restoring_method(ytrain)}


def get_restored_predictions(dictionary):
    return dictionary['test_predictions'], dictionary['train_predictions'], dictionary['ytrain']
