import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from utils.dataManagers.dataWrangler import *


def predict_with_kfold(results_class, model, param_grid, X, y, test, folds=5, yield_progress=True):
    kf = KFold(n_splits=folds, random_state=None, shuffle=False)
    errs = []
    res = []
    for train_index, test_index in kf.split(X):
        if yield_progress:
            print(f'iteration: {len(errs)} for class {type(model)}')
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        results = predict(results_class, model, param_grid, x_train,
                          np.log1p(y_train), test, name=None,
                          predictions_form_restoring_method=restore_predictions_from_log1p,
                          plot_results=False)
        errs.append(mean_squared_log_error(results.grid.predict(x_test), np.log1p(y_test)))
        res.append(results)
    errors = pd.Series(np.sqrt(errs))
    best_result_indx = errors.argmin()
    best_result = res[best_result_indx]
    best_result.refit(X, y)
    best_result.plot_results()
    if yield_progress:
        print(f'Best test error for model: {type(model)} is: {errors.min()}')
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


def predict(results_class, clf, param_grid, xtrain, ytrain, xtest, name,
            plot_results=True, store_classifier=False, store_predictions=True,
            predictions_form_restoring_method=None, n_jobs=-1, cv=5):
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
