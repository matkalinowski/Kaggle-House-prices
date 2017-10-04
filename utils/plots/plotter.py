import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns


class ResultsPlotter(object):
    def __init__(self, results):
        self.results = results

    def plot_train_vs_test_score_for_linear_clf(self, yscale='linear'):
        alphas = self.results.grid.param_grid['alpha']
        fig, ax = plt.subplots()
        best_param = self.results.grid.best_params_['alpha']
        train_score = self.results.grid.cv_results_['mean_train_score']
        test_score = self.results.grid.cv_results_['mean_test_score']

        ax.plot(alphas, np.sqrt(-train_score), label='train score')
        ax.plot(alphas, np.sqrt(-test_score), label='cross val score')
        plt.scatter(alphas, np.sqrt(-train_score))
        plt.scatter(alphas, np.sqrt(-test_score))
        plt.axvline(x=best_param, color='r', label=f'Choosed parameter alpha= {best_param}')
        plt.legend()
        plt.title('Train vs test scores error.')
        plt.xlabel('Alpha values')
        plt.ylabel('Root mean squared log error')
        plt.yscale(yscale)

    def plot_actual_vs_predicted_train_scores(self):
        fig, ax = plt.subplots()
        sns.regplot(self.results.train_predictions, self.results.ytrain, ax=ax)
        plt.title('Actual vs predicted train scores.')
        plt.xlabel('Predictions')
        plt.ylabel('Actual')

    def plot_best_predictors(self, predictors_count=10, tick_spacing=7000):
        best_cols = self.results.get_most_important_columns(predictors_count)
        best = self.results.coefficients[best_cols]

        fig, ax = plt.subplots(figsize=[9, 9])
        sns.barplot(best, best.index, ax=ax)
        ax.set_axisbelow(False)
        ax.xaxis.grid(color='w', linestyle='solid')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    def plot_results_distplot(self, bins=None):
        fig, ax = plt.subplots(nrows=3, figsize=[10, 6], tight_layout=True)
        sns.distplot(self.results.train_predictions, ax=ax[0], bins=bins)
        ax[0].set_title('Predictions.')
        sns.distplot(self.results.ytrain, ax=ax[1], bins=bins)
        ax[1].set_title('Actual.')
        sns.distplot(self.results.ytrain, ax=ax[2], bins=bins)
        sns.distplot(self.results.train_predictions, ax=ax[2], bins=bins)
        ax[2].set_title('Actual and predictions.')
