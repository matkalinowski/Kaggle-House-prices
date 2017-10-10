import datetime

import numpy as np
import pandas as pd
from sklearn.externals import joblib

from utils.plots.plotter import ResultsPlotter


def get_most_important_columns(series, columns_count=10):
    return list(np.abs(series).sort_values()[-columns_count:].index)


def get_excluded_columns(series):
    return list(series[series == 0].index)


def save_classifier(classifier, clf_name=None, clf_folder='classifiers/'):
    if clf_name is None:
        started = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
        clf_name = f'classifier-{started}'
    joblib.dump(classifier, clf_folder + clf_name + '.pkl', compress=9)


def get_predictions_df(predictions):
    df = pd.DataFrame(predictions, columns=['SalePrice'], index=range(1461, 2920))
    df.index.name = 'Id'
    return df


def get_submission_from_predictions(predictions, predictions_file_name=None):
    df = get_predictions_df(predictions)
    if predictions_file_name:
        df.to_csv(f'scores/{predictions_file_name}.csv')
    return df


class ClassifierResults(object):
    def __init__(self, grid, name, columns, train_predictions, test_predictions, ytrain, store_classifier=None,
                 store_predictions=None):
        self.coefficients = None
        self.grid = grid
        self.parameters = grid.best_params_
        self.train_predictions = train_predictions
        self.test_predictions = get_predictions_df(test_predictions)
        self.ytrain = ytrain
        self.columns = columns
        self.clf = grid.best_estimator_

        self.name = name
        if store_classifier:
            save_classifier(self.clf, self.name)
        if store_predictions:
            self.test_predictions.to_csv(f'scores/{self.name}.csv')

    def __repr__(self):
        return f'Results of {type(self.clf)} with given name: {self.name}.'

    def refit(self, X, y):
        self.ytrain = y
        self.grid.fit(X, y)
        self.clf = self.grid.best_estimator_
        self.train_predictions = self.grid.predict(X)

    def get_most_important_columns(self, columns_count=10):
        if self.coefficients is None:
            raise AttributeError('self.coefficients attribute should be initialized in children class')
        return get_most_important_columns(self.coefficients, columns_count)

    def get_excluded_columns(self):
        if self.coefficients is None:
            raise AttributeError('self.coefficients attribute should be initialized in children class')
        return get_excluded_columns(self.coefficients)

    def store_classifier(self):
        save_classifier(self.clf, self.name)

    def plot_results(self, plot_best_predictors=True, plot_actual_vs_predicted_test=True,
                     plot_residuals_for_train_data_set=True, plot_results_distplot=True, plot_train_vs_test=True,
                     results_distplot_bins=None, values_transformation=None):
        results_plotter = ResultsPlotter(self)
        if plot_best_predictors:
            results_plotter.plot_best_predictors(predictors_count=20)
        if plot_actual_vs_predicted_test:
            results_plotter.plot_actual_vs_predicted_train_scores()
        if plot_residuals_for_train_data_set:
            results_plotter.plot_residuals_for_train_data_set(values_transformation)
        if plot_results_distplot:
            results_plotter.plot_results_distplot(bins=results_distplot_bins)
        if plot_train_vs_test:
            results_plotter.plot_multiple_parameters_train_vs_test()


class RegressionResults(ClassifierResults):
    def __init__(self, grid, name, columns, train_predictions, test_predictions, ytrain, store_classifier=None,
                 store_predictions=None):
        super().__init__(grid, name, columns, train_predictions, test_predictions, ytrain, store_classifier,
                         store_predictions)
        self.coefficients = pd.Series(self.clf.coef_, index=self.columns)


class TreeResults(ClassifierResults):
    def __init__(self, grid, name, columns, train_predictions, test_predictions, ytrain, store_classifier=None,
                 store_predictions=None):
        super().__init__(grid, name, columns, train_predictions, test_predictions, ytrain, store_classifier,
                         store_predictions)
        self.coefficients = pd.Series(self.clf.feature_importances_, index=self.columns)
