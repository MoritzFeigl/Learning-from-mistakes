import statsmodels.stats.outliers_influence as out
import statsmodels.api as api
import statsmodels.tools as tools
import sklearn
import pandas as pd
import bayes_opt
import random
from xgboost import XGBRegressor
import json
import pprint
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class RegressionModel:

    def __init__(self, x, y):
        self.x = tools.tools.add_constant(x)
        self.y = y

    def variance_inflation(self):
        """ Computes variance inflation factor for regressors """
        vif = pd.DataFrame({'variables': self.x.columns,
                            'VIF': [out.variance_inflation_factor(self.x.values, i) for i in range(self.x.shape[1])]})
        print(vif)

    def center_data(self):
        """ Centers data to reduce influence of multicollinearity """
        self.x = self.x.drop(columns='const')
        data_centered = pd.DataFrame(sklearn.preprocessing.scale(self.x, with_mean='True', with_std='False'))
        data_centered.columns = self.x.columns
        data_centered.index = self.x.index
        self.x = data_centered
        self.x = tools.tools.add_constant(self.x)
        print("All columns successfully centered!")

    def fit(self):
        """ Fits a OLS regression model of the form y ~ x + intercept"""
        model = api.OLS(self.y, self.x)
        results = model.fit()
        print(results.summary())
        with open('results/tables/regression_model.csv', 'w') as fh:
            fh.write(results.summary().as_csv())
        print("Saved model summary in results/tables/regression_model.csv")


def create_lags(x, lag_variables, lag, drop_na=True):
    """ Builds a new DataFrame with additional lagged features """
    lag_x = x[lag_variables]
    if type(lag_x) is pd.DataFrame:
        new_dict = {}
        for col_name in lag_x:
            new_dict[col_name] = lag_x[col_name]
            # create lagged Series
            for l in range(1, lag + 1):
                new_dict['%s_lag%d' % (col_name, l)] = lag_x[col_name].shift(l)
        lagged_x = pd.DataFrame(new_dict, index=lag_x.index)
    else:
        the_range = range(lag + 1)
        lagged_x = pd.concat([lag_x.shift(i) for i in the_range], axis=1)
        lagged_x.columns = ['lag_%d' % i for i in the_range]
    x_without_lag_vars = x.drop(columns=lag_variables)
    x_with_lags = pd.concat([x_without_lag_vars, lagged_x], axis=1)
    if drop_na:
        x_with_lags = x_with_lags.dropna()
    return (x_with_lags)


# Functions for Bayesian hyperparameter optimization of XGBoost
def xgboost_cv(max_depth, gamma, min_child_weight, scale_pos_weight, n_estimators,
               reg_alpha, reg_lambda, max_delta_step, subsample,
               colsample_bytree, learning_rate, data, targets, n_jobs):
    """ XGBoost 5times repeated 5 fold cross validation.
    Inputs: XGBoost parameters, target labels and feature data
    Output: CV mean
    """
    random.seed(42)
    estimator = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=max_depth,
        gamma=gamma, min_child_weight=min_child_weight,
        scale_pos_weight=scale_pos_weight, reg_alpha=reg_alpha,
        reg_lambda=reg_lambda, max_delta_step=max_delta_step,
        subsample=subsample, colsample_bytree=colsample_bytree,
        learning_rate=learning_rate, n_jobs=n_jobs
    )
    rkf = sklearn.model_selection.RepeatedKFold(n_splits=5, n_repeats=5, random_state=1234)

    cval = sklearn.model_selection.cross_val_score(estimator, data, targets, cv=rkf,
                                                   scoring='neg_root_mean_squared_error')
    return cval.mean()


def optimize_xgboost(data, targets, init_points, n_iter, n_jobs):
    """ Apply Bayesian Optimization to XGBoost parameters and optimized parameters
    Inputs: feature data, label targets, number of random chosen initial points, number of iterations, number of cpus
    """

    def xgboost_crossval(max_depth, gamma, n_estimators,
                         min_child_weight, scale_pos_weight,
                         reg_alpha, reg_lambda,
                         max_delta_step, subsample,
                         colsample_bytree, learning_rate):
        """Wrapper of XGBoost cross validation."""
        return xgboost_cv(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            gamma=gamma, min_child_weight=min_child_weight,
            scale_pos_weight=scale_pos_weight, reg_alpha=reg_alpha,
            reg_lambda=reg_lambda, max_delta_step=int(max_delta_step),
            subsample=subsample, colsample_bytree=colsample_bytree,
            learning_rate=learning_rate, data=data, targets=targets, n_jobs=n_jobs
        )

    random.seed(42)
    optimizer = bayes_opt.BayesianOptimization(
        f=xgboost_crossval,
        pbounds=dict(
            n_estimators=(50, 5000),
            max_depth=(3, 20),
            gamma=(0.01, 5),
            min_child_weight=(0, 10),
            scale_pos_weight=(1.2, 5),
            reg_alpha=(4.0, 10.0),
            reg_lambda=(1.0, 10.0),
            max_delta_step=(0, 5),
            subsample=(0.5, 1.0),
            colsample_bytree=(0.3, 1.0),
            learning_rate=(0.0, 1.0)
        ),
        random_state=1234,
        verbose=2
    )
    random.seed(42)
    optimizer.maximize(n_iter=n_iter, init_points=init_points, acq="ucb", kappa=5)
    print('Maximum Value: {}'.format(optimizer.max['target']))
    print('Best Parameters:')
    print(optimizer.max['params'])
    return optimizer


class XGBoost:

    def __init__(self, x_train, y_train, x_val, y_val, x, y, data):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x = x
        self.y = y
        self.data = data

    def create_lagged_features(self, lag_variables, lag):
        """ Creates lagged features for training and validation features """
        self.x_train = create_lags(self.x_train, lag_variables, lag)
        self.x_val = create_lags(self.x_val, lag_variables, lag)
        self.x = create_lags(self.x, lag_variables, lag)
        self.y_train = self.y_train.iloc[lag:]
        self.y_val = self.y_val.iloc[lag:]
        self.y = self.y.iloc[lag:]
        self.data = self.data.iloc[lag:]

        print("Lagged features where added successfully!")

    def hyperpar_optimization(self, init_points, n_iter, n_jobs, model_run_name):
        """ Applies bayesian hyperparameter optimization for XGBoost parameters """
        random.seed(42)
        xgboost_opt = optimize_xgboost(self.x_train, self.y_train,
                                       init_points, n_iter, n_jobs)
        # save best model parameters
        with open('models/' + model_run_name + '_optimized_parameter.txt', 'w') as json_file:
            json.dump(xgboost_opt.max, json_file)
        print("Optimized model parameters are saved in models/" + model_run_name + "_optimized_parameter.txt")

        # save full hyperparameter optimization log
        hyperpar_opt_results = list()
        for i, iteration in enumerate(xgboost_opt.res):
            iteration_results_unformated = pd.DataFrame(iteration).transpose()
            iteration_results = iteration_results_unformated.iloc[1]
            iteration_results['RMSE'] = -iteration_results_unformated.iloc[0, 0]
            hyperpar_opt_results.append(iteration_results.to_dict())
        pd.DataFrame(hyperpar_opt_results).to_csv('models/' + model_run_name + '_hyperpar_search.csv')
        print("Fully hyperparameter search results are saved in models/" + model_run_name + "_hyperpar_search.csv")

    def fit(self, model_run_name):
        """ Fit XGBoost model with previously derived optimal hyperparameters """
        with open('models/' + model_run_name + '_optimized_parameter.txt') as f:
            optimized_parameters = json.load(f)
        parameters = {
            'objective': "reg:squarederror",
            'n_estimators': int(optimized_parameters['params']['n_estimators']),
            'learning_rate': optimized_parameters['params']['learning_rate'],
            'max_depth': int(optimized_parameters['params']['max_depth']),
            'gamma': optimized_parameters['params']['gamma'],
            'min_child_weight': optimized_parameters['params']['min_child_weight'],
            'max_delta_step': int(optimized_parameters['params']['max_delta_step']),
            'subsample': optimized_parameters['params']['subsample'],
            'colsample_bytree': optimized_parameters['params']['colsample_bytree'],
            'scale_pos_weight': optimized_parameters['params']['scale_pos_weight'],
            'reg_alpha': optimized_parameters['params']['reg_alpha'],
            'reg_lambda': optimized_parameters['params']['reg_lambda']
        }
        print("Optimized parameters:")
        pprint.pprint(parameters)
        self.model = XGBRegressor(**parameters)
        random.seed(42)
        self.model.fit(self.x_train, self.y_train)
        prediction = self.model.predict(self.x_val)
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(self.y_val, prediction))
        print("Optimized model RMSE in test set: %f" % rmse)

    def plot_prediction(self,  model_run_name):
        prediction = pd.DataFrame({"prediction": self.model.predict(self.x)})
        prediction.index = self.x.index

        xbgoost_predicted_wt = self.data['wt_predicted_point_640'] + prediction["prediction"]
        plot_data = pd.concat(
            [self.data[['wt_observed_point_640', 'wt_predicted_point_640', 'calibration_validation']],
             xbgoost_predicted_wt], axis=1)
        sns.set(rc={'figure.figsize': (18, 9)})
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.5)
        plot = plot_data.plot(linewidth=2, legend=False)
        plot.legend(title='', loc='upper right',
                    labels=['Observed', 'Model predicted', 'Model + residual prediction'])
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(xbgoost_predicted_wt, plot_data['wt_observed_point_640']))
        plot.set_title('Predicted and observed stream water temperature, RMSE: %f' % (rmse))
        plot.figure.savefig('results/figures/06_XGBoost_prediction_' + model_run_name + '.png')

    def shap(self):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.x)