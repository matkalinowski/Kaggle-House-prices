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
        self.grid = grid
        self.parameters = grid.best_params_
        self.train_predictions = train_predictions
        self.test_predictions = get_predictions_df(test_predictions)
        self.ytrain = ytrain
        self.columns = columns
        self.clf = grid.best_estimator_
        self.name = name
        self.coefficients = pd.Series(self.clf.coef_, index=self.columns)
        if store_classifier:
            save_classifier(self.clf, self.name)
        if store_predictions:
            self.test_predictions.to_csv(f'scores/{self.name}.csv')

    def get_most_important_columns(self, columns_count=10):
        return get_most_important_columns(self.coefficients, columns_count)

    def get_excluded_columns(self):
        return get_excluded_columns(self.coefficients)

    def store_classifier(self):
        save_classifier(self.clf, self.name)

    def plot_results(self, plot_best_predictors=True, plot_train_vs_test_for_linear_clf=True,
                     plot_actual_vs_predicted_test=True,
                     plot_results_distplot=True, results_distplot_bins=None):

        pp = ResultsPlotter(self)
        if plot_best_predictors:
            pp.plot_best_predictors(predictors_count=20)
        if plot_train_vs_test_for_linear_clf:
                pp.plot_train_vs_test_score_for_linear_clf()
        if plot_actual_vs_predicted_test:
            pp.plot_actual_vs_predicted_train_scores()
        if plot_results_distplot:
            pp.plot_results_distplot(bins=results_distplot_bins)
