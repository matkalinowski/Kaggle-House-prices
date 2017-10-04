import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from utils.ClassifierResults import ClassifierResults
from utils.dataManagers.dataWrangler import *
from utils.plots.plotter import PredictionsPlotter


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


def predict(clf, param_grid, xtrain, ytrain, xtest, name,
            store_classifier=False, store_predictions=True,
            plot_best_predictors=True, plot_train_vs_test=True,
            plot_actual_vs_predicted_test=True,
            plot_results_distplot=True, results_distplot_bins=None,
            predictions_form_restoring_method=None):
    grid = GridSearchCV(clf, param_grid, scoring='neg_mean_squared_log_error')
    grid.fit(xtrain, ytrain)

    test_predictions = grid.predict(xtest)
    train_predictions = grid.predict(xtrain)
    if predictions_form_restoring_method is not None:
        test_predictions = predictions_form_restoring_method(test_predictions)
        train_predictions = predictions_form_restoring_method(train_predictions)
        ytrain = predictions_form_restoring_method(ytrain)

    results = ClassifierResults(grid, name, xtest.columns,
                                test_predictions, store_classifier=store_classifier,
                                store_predictions=store_predictions)

    pp = PredictionsPlotter(grid)
    if plot_best_predictors:
        pp.plot_best_predictors(results, 20)
    if plot_train_vs_test:
        pp.plot_train_vs_test_score_for_linear_clf(param_grid['alpha'])
    if plot_actual_vs_predicted_test:
        pp.plot_actual_vs_predicted_train_scores(train_predictions, ytrain)
    if plot_results_distplot:
        pp.plot_results_distplot(train_predictions, ytrain, np.log1p, bins=results_distplot_bins)
    return results, grid
