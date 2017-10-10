import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec


def createQuery(best_params):
    query = ''
    for p in best_params:
        if type(best_params[p]) == str:
            query += f'({p} == \'{best_params[p]}\') and '
        else:
            query += f'({p} == {best_params[p]}) and '
    return query.replace('None', '-1')[:-5]


def plot_residuals(diff, title):
    fig = plt.figure(figsize=[10, 6])
    gs = gridspec.GridSpec(nrows=2, ncols=3, width_ratios=[2, 1, 1])
    ax = plt.subplot(gs[:, :-1])

    sns.regplot(diff.index.values, diff, ax=ax)
    diff_mean = diff.mean()
    diff_std = diff.std()
    outliers = diff[(diff > diff_mean + 3 * diff_std) | (diff < diff_mean - 3 * diff_std)]

    ax.fill_between(diff.index.values, diff_mean + 2 * diff_std, diff_mean - 2 * diff_std, alpha=.4,
                    label='2 std of the mean')
    ax.fill_between(diff.index.values, diff_mean + 3 * diff_std, diff_mean - 3 * diff_std, alpha=.2,
                    label='3 std of the mean')
    diff.describe().round(5)
    ax.legend()
    ax.set_title(title)
    ax.grid(color='black', linestyle='-', linewidth=.1)

    ax = plt.subplot(gs[0, -1])
    ax.axis('off')
    diff_info = diff.describe()[['mean', 'std', '50%', 'min', 'max']].round(5).to_frame()
    table = ax.table(cellText=diff_info.values.astype('str'), rowLabels=diff_info.index.values, loc='right')
    table.set_fontsize(12)

    ax = plt.subplot(gs[1, -1])
    ax.axis('off')
    ax.text(.05, .9, f'Number of outliers = {outliers.count()}\nOutliers ids: {outliers.index.values}', fontsize=12)


class ResultsPlotter(object):
    def __init__(self, results):
        self.results = results

    def plot_actual_vs_predicted_train_scores(self):
        fig, ax = plt.subplots()
        sns.regplot(self.results.train_predictions, self.results.ytrain, ax=ax)
        plt.title('Actual vs predicted train scores.')
        plt.xlabel('Predictions')
        plt.ylabel('Actual')

    def plot_residuals_for_train_data_set(self, values_transformation=None):
        y = self.results.ytrain
        diff = pd.Series(self.results.train_predictions - y)
        plot_residuals(diff, 'Residual plot')

        if values_transformation is not None:
            y = self.results.ytrain
            preds = self.results.train_predictions

            diff = pd.Series(values_transformation(preds) - values_transformation(y))
            plot_residuals(diff, f'Residual plot with {values_transformation} method used.')

    def plot_best_predictors(self, predictors_count=10, tick_spacing=.01):
        best_cols = self.results.get_most_important_columns(predictors_count)
        best = self.results.coefficients[best_cols]

        fig, ax = plt.subplots(figsize=[9, 9])
        sns.barplot(best, best.index, ax=ax)
        ax.set_axisbelow(False)
        ax.xaxis.grid(color='w', linestyle='solid')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.xticks(rotation=90)

    def plot_results_distplot(self, bins=None):
        fig, ax = plt.subplots(nrows=3, figsize=[10, 6], tight_layout=True)
        sns.distplot(self.results.train_predictions, ax=ax[0], bins=bins)
        ax[0].set_title('Predictions.')
        sns.distplot(self.results.ytrain, ax=ax[1], bins=bins)
        ax[1].set_title('Actual.')
        sns.distplot(self.results.ytrain, ax=ax[2], bins=bins)
        sns.distplot(self.results.train_predictions, ax=ax[2], bins=bins)
        ax[2].set_title('Actual and predictions.')

    def plot_multiple_parameters_train_vs_test(self):
        grid = self.results.grid
        df = pd.DataFrame(grid.cv_results_['params'])
        df.fillna(-1, inplace=True)

        query = createQuery(grid.best_params_)
        best_params_index = df.query(query).index

        checked_params = [str(cv_result).replace('{', '').replace('}', '') for cv_result in grid.cv_results_['params']]
        fig, ax = plt.subplots(figsize=[10, 10], tight_layout=True)

        df['train_score'] = np.sqrt(-grid.cv_results_['mean_train_score'])
        df['test_score'] = np.sqrt(-grid.cv_results_['mean_test_score'])

        ax.scatter(x=range(len(checked_params)), y=df.test_score, label=None)
        ax.plot(df.test_score, label='cross val score')

        ax.scatter(x=range(len(checked_params)), y=df.train_score, label=None)
        ax.plot(df.train_score, label='train score')
        plt.axvline(x=best_params_index, color='r', label=f'Choosed parameters: {grid.best_params_}')

        locator = ticker.MultipleLocator()
        ax.xaxis.set_major_locator(locator)
        ax.set_xticks(np.arange(len(checked_params)))
        ax.set_xticklabels(checked_params, rotation=90)
        ax.grid(color='black', linestyle='-', linewidth=.1)

        plt.legend()
        plt.title('Train vs test scores error.')
        plt.xlabel('Parameter values')
        plt.ylabel('Root mean squared log error')
