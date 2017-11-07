### Informations

This repository is my solution for [kaggle housing prices competiton](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

In project creation I tried to create easiest way to ensemble scikit-learn models, visually inspect possible problems and automate most of the process needed to obtain best model.

If you want to take a quick peek at results of the project look at [Ensemble](https://github.com/matkalinowski/Kaggle-House-prices/blob/master/Ensemble.ipynb) file.

Visual analysis can be found in [Opening Analysis](https://github.com/matkalinowski/Kaggle-House-prices/blob/master/Opening%20Analysis.ipynb) file.

### Further work
My scores are lower than submission results, therefore I have overfitting problem due to high variance. Thus trying smaller set of features or better selection should work well.

My models seem to be overfitting right now also due to parameters I choosed, so better hyper parameter tuning is also considered as one of possible improvement part.

* Feature engineering:
    * especially on part of data set that got excluded, it still may contain some amount information,
    * some of the numerical features are categorical- change their type on category may have sense in some cases,
    * create simplifications of some of the categorical features witch are under sampled,
    * create combinations of existing features,
    * create polynomials of some numerical features.
        * Checking for non linear relationships first may be a good idea.
    * Use [box cox](http://onlinestatbook.com/2/transformations/box-cox.html) instead of logging skewed features.
* Handle null values in different way. For example:
    * filling according to neighborhood,
    * look closely to every feature and decide what to do,
    * filling across whole data set (train + test)
* Try different models:
    * bayesian neural networks,
    * SVM with linear kernel,
    * xgboost with different estimators count in single estimator.
* Change weighting method.
    * According to features ranking.
    * According to minor value counts / major value counts for categoricals
        * for numerical features group them in bins of similar count.
    * Connection between two previous ideas,
    * based of features ranking.
* Add meta model
    * model that will create feature based on predictions of previous ones,
    * for more information use this [blog post](http://blog.kaggle.com/2017/06/15/stacking-made-easy-an-introduction-to-stacknet-by-competitions-grandmaster-marios-michailidis-kazanova/)
* Look closely to outliers,
    * maybe I substracted too much or to little samples.
* Change kfold and cv folds number.
* Manually check differences between samples that are classified as outliers.

Ensembler:
* Find way to store predictions and predictors in database or in file system,
* visually compare models between each other.