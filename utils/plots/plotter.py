import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns


class PredictionsPlotter(object):
    def __init__(self, grid):
        self.grid = grid

    def plot_train_vs_test_score_for_linear_clf(self, alphas, yscale='linear'):
        fig, ax = plt.subplots()
        best_param = self.grid.best_params_['alpha']
        train_score = self.grid.cv_results_['mean_train_score']
        test_score = self.grid.cv_results_['mean_test_score']

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

    def plot_actual_vs_predicted_train_scores(self, predictions, ytrain):
        fig, ax = plt.subplots()
        sns.regplot(predictions, ytrain, ax=ax)
        plt.title('Actual vs predicted train scores.')
        plt.xlabel('Predictions')
        plt.ylabel('Actual')

    @staticmethod
    def plot_best_predictors(results, predictors_count=10, tick_spacing=7000):
        best_cols = results.get_most_important_columns(predictors_count)
        best = results.coefficients[best_cols]

        fig, ax = plt.subplots(figsize=[9, 9])
        sns.barplot(best, best.index, ax=ax)
        ax.set_axisbelow(False)
        ax.xaxis.grid(color='w', linestyle='solid')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    @staticmethod
    def plot_results_distplot(predictions, ytrain, transformation_to_visualize=None, bins=None):
        if transformation_to_visualize is not None:
            predictions = transformation_to_visualize(predictions)
            ytrain = transformation_to_visualize(ytrain)

        fig, ax = plt.subplots(nrows=3, figsize=[10, 6], tight_layout=True)
        sns.distplot(predictions, ax=ax[0], bins=bins)
        ax[0].set_title('Predictions.')
        sns.distplot(ytrain, ax=ax[1], bins=bins)
        ax[1].set_title('Actual.')
        sns.distplot(ytrain, ax=ax[2], bins=bins)
        sns.distplot(predictions, ax=ax[2], bins=bins)
        ax[2].set_title('Actual and predictions.')
