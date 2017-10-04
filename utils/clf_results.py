import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.externals import joblib


def get_most_important_columns(series, columns_count=10):
    return list(np.abs(series).sort_values()[-columns_count:].index)


def get_excluded_columns(series):
    return list(series[series == 0].index)


def save_classifier(classifier, clf_name=None, clf_folder='classifiers/'):
    if clf_name is None:
        started = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
        clf_name = f'classifier-{started}'
    joblib.dump(classifier, clf_folder + clf_name + '.pkl', compress=9)


def plot_best_predictors(best, tick_spacing=7000):
    fig, ax = plt.subplots(figsize=[9, 9])
    sns.barplot(best, best.index, ax=ax)
    ax.set_axisbelow(False)
    ax.xaxis.grid(color='w', linestyle='solid')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))


def get_predictions_df(predictions):
    df = pd.DataFrame(predictions, columns=['SalePrice'], index=range(1461, 2920))
    df.index.name = 'Id'
    return df


class clfResult(object):
    def __init__(self, clf, name, columns, predictions, parameters, store_classifier=None, store_predictions=None):
        self.parameters = parameters
        self.predictions = get_predictions_df(predictions)
        self.columns = columns
        self.clf = clf
        self.name = name
        self.coefficients = pd.Series(self.clf.coef_, index=self.columns)
        if store_classifier:
            save_classifier(self.clf, self.name)
        if store_predictions:
            self.predictions.to_csv(f'scores/{self.name}.csv')

    def get_most_important_columns(self, columns_count=10):
        return get_most_important_columns(self.coefficients, columns_count)

    def get_excluded_columns(self):
        return get_excluded_columns(self.coefficients)

    def store_classifier(self):
        save_classifier(self.clf, self.name)

    def plot_best_predictors(self, predictors_count=10, tick_spacing=7000):
        best_cols = self.get_most_important_columns(predictors_count)
        plot_best_predictors(self.coefficients[best_cols], tick_spacing)
