import pickle

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from utils.ClassifierResults import *
from utils.dataManagers.dataWrangler import *
from utils.plots.plotter import get_scores_df_from_grid

SCORES_FOLDER = 'scores'


def get_test_predictions_df(predictions):
    df = pd.DataFrame(predictions, columns=['SalePrice'], index=range(1461, 2920))
    df.index.name = 'Id'
    return df


def get_submission_from_predictions(predictions, predictions_file_name=None):
    df = get_test_predictions_df(predictions)
    if predictions_file_name:
        df.to_csv(f'{SCORES_FOLDER}/{predictions_file_name}.csv')
    return df


def store_object(clf, name):
    pickle.dump(clf, open(f'{name}.p', 'wb'))


class RegressionModel(object):
    def __init__(self, results_class, estimator, param_grid, name, n_iter=None, weights=None):
        """
        Custom wrapper over regression models.
        :param results_class: Results class @see ClassifierResults.py
            example: use RegressionResults for Lasso, Ridge, TreeResults for xgb, random forests.
        :param estimator: Estimator for example XGBRegressor, Lasso...
        :param param_grid: parameters to be fitted on estimator,
        :param name: name of model,
        :param n_iter:  count of iterations on randomized search in every kfold.
        :param weights: weights used for tree methods.
        """
        self.n_iter = n_iter
        self.name = name
        self.weights = weights
        self.param_grid = param_grid
        self.estimator = estimator
        self.results_class = results_class
        self.cv_fit_results = []

    def get_min_val_error(self):
        df = pd.DataFrame(self.cv_fit_results)
        return df.val_err.min()

    def get_best_cv_estimator_grid(self):
        df = pd.DataFrame(self.cv_fit_results)
        return df.iloc[df.val_err.argmin()].grid, df.val_err.min()


class EnsemblerNotFittedError(Exception):
    def __init__(self, message):
        super(self).__init__(message)


class Ensembler(object):
    def __init__(self, models, name):
        """
        Ensembler class containing models.
        After fitting:
            kfold_splits_indexes will contain indexes of kfold splits,
            best_grids will contain best grid for each model,
        After using calculate_results_objects:
            results will contain results customized classes- use it to plot results and gain easy access to test
            predictions @See ClassifierResults.py.
        :param models: Models list.
        :param name: Name of ensembler.
        """
        self.name = name
        self.models = models
        self.best_grids = []
        self.results = []
        self.kfold_splits_indexes = []

    @staticmethod
    def _get_results_dict(split_counter, val_err, grid):
        result = dict()
        result['kf_split'] = split_counter
        result['val_err'] = val_err
        result['grid'] = grid
        return result

    @staticmethod
    def _get_kfold_sets(x_train, y_train, train_index, test_index):
        return x_train.iloc[train_index], x_train.iloc[test_index], y_train.iloc[train_index], y_train.iloc[test_index]

    def fit(self, x_train, y_train, k_folds=5, cv_folds=5, n_jobs=-1, yield_progress=True):
        """
        Fitting data on every model given in constructor.
        :param x_train: Train data.
        :param y_train: Train response.
        :param k_folds: Number of k_folds.
        :param cv_folds: Number of gridSearch or randomSearch folds.
        :param n_jobs: Used in gridSearch or randomSearch. (Number of jobs to run in parallel.)
        :param yield_progress: Print progress information.
        :return: None. To see results use method ''predict'' to see test results only or ''calculate_results_objects''.
        Second one is recommended.
        """
        start_time = datetime.datetime.now()
        kf = KFold(n_splits=k_folds, random_state=None, shuffle=False)
        split_counter = 0
        for train_index, test_index in kf.split(x_train):
            lap_start = datetime.datetime.now()
            self.kfold_splits_indexes.append([train_index, test_index])
            x_train_kf, x_val_kf, y_train_kf, y_val_kf = self._get_kfold_sets(x_train, y_train, train_index, test_index)

            for model in self.models:
                grid, val_err = self._fit_single_model(cv_folds, model, n_jobs, x_train_kf, x_val_kf,
                                                       y_train_kf,
                                                       y_val_kf,
                                                       yield_progress, split_counter)
                get_scores_df_from_grid(grid, store=True)
                model.cv_fit_results.append(self._get_results_dict(split_counter, val_err, grid))
            lap_end = datetime.datetime.now()
            print(f'kfold lap started at: {lap_start} and ended at: {lap_end}, took: {lap_end - lap_start}\n---')
            split_counter += 1

        for model in self.models:
            self._refit_trained_model(model, x_train, y_train, yield_progress)

        self._print_calculations_time(yield_progress, start_time)

    def _refit_trained_model(self, model, x_train, y_train, yield_progress):
        grid, val_err = model.get_best_cv_estimator_grid()
        if yield_progress:
            print(f'Refitting model {model.name}')
            print(f'Best validation error for model: {model.name} is: {val_err},'
                  f' chosen parameters are: {grid.best_params_}')

        Ensembler._fit_grid(grid, model, x_train, y_train)
        self.best_grids.append(grid)
        return grid

    @staticmethod
    def _fit_single_model(gridSearch_folds, model, n_jobs, x_train_kf, x_val_kf, y_train_kf, y_val_kf,
                          yield_progress,
                          split_counter):
        if yield_progress:
            print(f'kfold number: {split_counter} for model: {model.name},'
                  f' time is: {datetime.datetime.now()}')
        if model.n_iter:
            print('Performing RandomizedSearchCV.')
            grid = RandomizedSearchCV(model.estimator, model.param_grid, cv=gridSearch_folds,
                                      n_iter=model.n_iter,
                                      scoring='neg_mean_squared_error',
                                      n_jobs=n_jobs)

        else:
            print('Performing GridSearchCV.')
            grid = GridSearchCV(model.estimator, model.param_grid, cv=gridSearch_folds,
                                scoring='neg_mean_squared_error',
                                n_jobs=n_jobs)

        Ensembler._fit_grid(grid, model, x_train_kf, y_train_kf)
        val_err = mean_squared_error(np.expm1(y_val_kf), np.expm1(grid.predict(x_val_kf)))
        if yield_progress:
            print(f'Current mean_squared_error on validation set(logged pred and output) is: {val_err}')
        return grid, val_err

    @staticmethod
    def _fit_grid(grid, model, x, y):
        if model.weights is not None:
            grid.fit(x, y, sample_weight=model.weights)
        else:
            grid.fit(x, y)

    @staticmethod
    def _get_predictions(grid, data, predictions_form_restoring_method=None):
        if predictions_form_restoring_method is not None:
            return predictions_form_restoring_method(grid.predict(data))
        else:
            return grid.predict(data)

    @staticmethod
    def _print_calculations_time(yield_progress, start):
        end = datetime.datetime.now()
        if yield_progress:
            print(f'\nCalculations started at: {start} and ended at {end}, took: {end-start}')

    def predict(self, x_test, predictions_form_restoring_method=None):
        results = []
        for grid in self.best_grids:
            results.append(
                get_test_predictions_df(self._get_predictions(grid, x_test, predictions_form_restoring_method)))
        return results

    def calculate_results_objects(self, x_train, y_train, x_test, plot_best_results=True, store_predictions_score=None,
                                  predictions_form_restoring_method=None, store_classifier=None):
        self.results = []
        if self.best_grids is None:
            raise EnsemblerNotFittedError('Perform fit operation on ensembler first.')

        for i, grid in enumerate(self.best_grids):
            model = self.models[i]
            test_predictions = get_test_predictions_df(grid.predict(x_test))
            train_predictions = pd.Series(grid.predict(x_train), index=x_train.index)
            name = model.name + '_from(' + self.name + ')'

            restored_data = None
            if predictions_form_restoring_method is not None:
                restored_data = RestoredData(test_predictions,
                                             train_predictions, y_train,
                                             predictions_form_restoring_method)
            res = model.results_class(grid, name, x_test.columns,
                                      train_predictions, test_predictions, y_train, model.get_min_val_error(),
                                      restored_data,
                                      store_classifier=store_classifier,
                                      store_predictions=store_predictions_score)
            if plot_best_results:
                res.plot_results()
            self.results.append(res)
        return self.results

    def ensemble_predictions(self, wages, store=True):
        if self.results is None:
            raise EnsemblerNotFittedError('Calculate results objects on ensembler first.')
        result = 0
        for i, w in enumerate(wages):
            res = self.results[i]
            if res.restored_data:
                result += res.restored_data.test_predictions * w
            else:
                result += res.test_predictions * w
        if store:
            result.to_csv(f'{SCORES_FOLDER}/{self.name}_wages({wages}).csv')
        return result


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


class RestoredData(object):
    def __init__(self, test_predictions, train_predictions, ytrain, restore_method):
        self.test_predictions = restore_method(test_predictions)
        self.train_predictions = restore_method(train_predictions)
        self.ytrain = restore_method(ytrain)

    def get_restored_values(self):
        return self.test_predictions, self.train_predictions, self.ytrain

def plot_with_y_presence_check(ax, y, plot_function, series):
    if y is not None:
        if len(series.index) != len(y.index):
            print(len(series.index))
            print(len(y.index))
            y = y.iloc[series.index]
            print(len(y.index))
        plot_function(series, y, ax=ax, color="#9b59b6")
    else:
        plot_function(series, ax=ax, color="#9b59b6")