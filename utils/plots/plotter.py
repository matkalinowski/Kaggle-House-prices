import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
import time


def createQuery(best_params):
    query = ''
    for p in best_params:
        if type(best_params[p]) == str:
            query += f'({p} == \'{best_params[p]}\') and '
        else:
            query += f'({p} == {best_params[p]}) and '
    return query.replace('None', '-1')[:-5]


def plot_confidence_interval(ax, index, mean, std, additional_label_info, std_count=2):
    ax.fill_between(index, mean - std_count * std, mean + std_count * std, alpha=std_count / 20,
                    label=f'{std_count}sd of score mean- {additional_label_info}')


def plot_residuals(actual, predicted, title):
    diff = pd.Series(actual - predicted)

    fig = plt.figure(figsize=[10, 6])
    gs = gridspec.GridSpec(nrows=2, ncols=3, width_ratios=[2, 1, 1])
    ax = fig.add_subplot(gs[:, :-1])

    sns.regplot(predicted, diff, ax=ax, scatter_kws={'alpha': 0.3}, line_kws={'alpha': 0.5})
    diff_mean = diff.mean()
    diff_std = diff.std()
    outliers = diff[(diff.values > diff_mean + 3 * diff_std) | (diff.values < diff_mean - 3 * diff_std)]

    conf_range = np.linspace(predicted.min(), predicted.max())
    plot_confidence_interval(ax, conf_range, diff_mean, diff_std, additional_label_info='train', std_count=2)
    plot_confidence_interval(ax, conf_range, diff_mean, diff_std, additional_label_info='train', std_count=3)

    ax.legend()
    ax.set_title(title)
    ax.set_ylabel('Residuals')
    ax.set_xlabel('Predicted sale price')
    ax.grid(color='black', linestyle='-', linewidth=.1)

    ax = fig.add_subplot(gs[0, -1])
    ax.axis('off')
    diff_info = diff.describe()[['mean', 'std', '50%', 'min', 'max']].round(5).to_frame()
    table = ax.table(cellText=diff_info.values.astype('str'), rowLabels=diff_info.index.values, loc='right')
    table.set_fontsize(12)

    ax = fig.add_subplot(gs[1, -1])
    ax.axis('off')
    ax.text(.05, .9, f'Number of outliers = {outliers.count()}\nOutliers ids: {outliers.index.values}', fontsize=12)

    return fig, outliers


def normal_probability(ax, predictions, label):
    ys = predictions.sort_values()
    n = len(ys)
    xs = np.random.normal(0, 1, n)
    xs.sort()
    ax.plot(xs, ys, label=label)
    plt.xlabel('Standard deviations from mean')
    plt.ylabel('Predictions')
    plt.title('Normal probability plot')
    plt.legend()


def plot_multiple_parameters_train_vs_test(grid, yscale='linear', plot_confidence_intervals=True):
    df = pd.DataFrame(grid.cv_results_['params'])
    df.fillna(-1, inplace=True)

    query = createQuery(grid.best_params_)
    best_params_index = df.query(query).index

    checked_params = [str(cv_result).replace('{', '').replace('}', '') for cv_result in grid.cv_results_['params']]
    width_multiplier = df.shape[0] // 60 + 1
    fig, ax = plt.subplots(figsize=[width_multiplier * 10, 10], tight_layout=True)

    df['mean_train_score'] = np.sqrt(-grid.cv_results_['mean_train_score'])
    df['mean_test_score'] = np.sqrt(-grid.cv_results_['mean_test_score'])
    df['std_train_score'] = np.sqrt(grid.cv_results_['std_train_score'])
    df['std_test_score'] = np.sqrt(grid.cv_results_['std_test_score'])

    timestr = time.strftime("%Y%m%d-%H%M%S")
    df.to_csv(f'results/grid_{timestr}.csv', index=None)

    if plot_confidence_intervals:
        plot_confidence_interval(ax, df.index.values, df.mean_test_score, df.std_test_score, std_count=2,
                                 additional_label_info='cv')
        plot_confidence_interval(ax, df.index.values, df.mean_test_score, df.std_test_score, std_count=3,
                                 additional_label_info='cv')

        plot_confidence_interval(ax, df.index.values, df.mean_train_score, df.std_train_score, std_count=2,
                                 additional_label_info='train')
        plot_confidence_interval(ax, df.index.values, df.mean_train_score, df.std_train_score, std_count=3,
                                 additional_label_info='train')

    ax.scatter(x=range(len(checked_params)), y=df.mean_test_score, label=None)
    ax.plot(df.mean_test_score, label='cross val score')

    ax.scatter(x=range(len(checked_params)), y=df.mean_train_score, label=None)
    ax.plot(df.mean_train_score, label='train score')
    plt.axvline(x=best_params_index, color='r', label=f'Choosed parameters: {grid.best_params_}')

    locator = ticker.MultipleLocator()
    ax.xaxis.set_major_locator(locator)
    ax.set_xticks(np.arange(len(checked_params)))
    ax.set_xticklabels(checked_params, rotation=90)
    ax.grid(color='black', linestyle='-', linewidth=.1)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Train vs test scores error.')
    plt.xlabel('Parameter values')
    plt.ylabel('Root mean squared log error')
    plt.yscale(yscale)


class ResultsPlotter(object):
    def __init__(self, results):
        self.results = results

    def plot_actual_vs_predicted_train_scores(self):
        fig, ax = plt.subplots(figsize=[8, 8])
        sns.regplot(self.results.train_predictions, self.results.ytrain, ax=ax, ci=99, label='results')
        ideal = np.linspace(self.results.ytrain.min(), self.results.ytrain.max())
        plt.plot(ideal, ideal, c='r', linewidth=.5, label='ideal values line')
        plt.title('Actual vs predicted train scores.')
        plt.xlabel('Predictions')
        plt.ylabel('Actual')
        plt.legend()

    def plot_residuals_for_train_data_set(self):
        y = self.results.ytrain
        plot_residuals(y, self.results.train_predictions, 'Residual plot')

        if self.results.restored_data is not None:
            _, train_predictions, y = self.results.restored_data.get_restored_values()
            plot_residuals(y, train_predictions, f'Residual plot with values restored.')

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

    def plot_normal_probability(self):
        results = self.results

        fig, ax = plt.subplots(figsize=[8, 8])
        normal_probability(ax, results.train_predictions, 'predicted')
        normal_probability(ax, results.ytrain, 'actual')

        if results.restored_data is not None:
            fig, ax = plt.subplots(figsize=[8, 8])
            _, predictions, actual = results.restored_data.get_restored_values()
            normal_probability(ax, predictions, 'predicted')
            normal_probability(ax, actual, 'actual')
