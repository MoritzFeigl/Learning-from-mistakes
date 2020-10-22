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
import shap
import os
from sklearn.cluster import AgglomerativeClustering
import matplotlib.image as img
import glob

class RegressionModel:

    def __init__(self, x, y, model_variables):
        self.x = tools.tools.add_constant(x[model_variables])
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
        self.model_variables = x_train.columns.tolist()
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x = x
        self.y = y
        self.data = data

    def hyperpar_optimization(self, init_points, n_iter, n_jobs, model_run_name):
        """ Applies bayesian hyperparameter optimization for XGBoost parameters """
        if os.path.isfile("models/" + model_run_name + "_optimized_parameter.txt"):
            print("Model was already optimized. Choose different model_run_name or load previous optimization results")
        else:
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

    def plot_prediction(self, model_run_name):
        prediction = pd.DataFrame({"prediction": self.model.predict(self.x)})
        prediction.index = self.x.index

        xbgoost_predicted_wt = self.data['wt_predicted_point_640'] + prediction["prediction"]
        plot_data = pd.concat(
            [self.data[['wt_observed_point_640', 'wt_predicted_point_640', 'calibration_validation']],
             xbgoost_predicted_wt], axis=1)
        plot_data.index = pd.to_datetime(plot_data.index)
        sns.set(rc={'figure.figsize': (24, 7)})
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.5)
        plot = plot_data.plot(linewidth=2, legend=False)
        plot.legend(title='', loc='upper right',
                    labels=['Observation', 'Model prediction', 'Model + residual prediction'])
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(xbgoost_predicted_wt, plot_data['wt_observed_point_640']))
        plot.set_title(f'Predicted and observed stream water temperature, RMSE:  {rmse:.4} °C')
        plot.figure.savefig('results/figures/06_XGBoost_prediction_' + model_run_name + '.png')

    def shap_values(self):
        explainer = shap.TreeExplainer(self.model)
        shap_array = explainer.shap_values(self.x)
        # aggregate lagged shap values
        self.shap_values = pd.DataFrame(shap_array)
        self.shap_values.columns = self.x.columns
        self.shap_values.index = self.data.index
        lag_variables = [x for x in self.model_variables if x not in ["sin_hour", "cos_hour"]]
        self.aggregated_shap_values = pd.DataFrame({"hour":self.shap_values[["sin_hour", "cos_hour"]].sum(axis=1)})
        for variable in lag_variables:
            self.aggregated_shap_values[variable] = self.shap_values.filter(regex=(variable + "*")).sum(axis=1)
        # aggregate hours
    def plot_variable_importance(self, model_variables_names, n_cluster):

        # define Cluster dummy columns
        cluster = AgglomerativeClustering(n_clusters=n_cluster, affinity='euclidean', linkage='ward')
        labels = cluster.fit_predict(self.aggregated_shap_values)
        dummies = pd.get_dummies(labels)
        dummies.index = self.y.index

        # get prediction
        prediction = pd.DataFrame({"prediction": self.model.predict(self.x)})
        prediction.index = self.data.index

        # define df for plotting
        plot_data = pd.concat(
            [prediction["prediction"], self.y, self.aggregated_shap_values, dummies], axis=1)
        plot_data.index = self.data.index
        plot_data_timesteps = pd.to_datetime(self.data.index).strftime("%Y-%m-%d")
        # plot limits for y-axis
        y_max = np.max([self.y.values.max(), prediction["prediction"].max()])
        y_min = np.min([self.y.values.min(), prediction["prediction"].min()])
        # plot in 2 day steps
        all_days = pd.to_datetime(self.data.index).strftime("%Y-%m-%d").unique()[::2]
        os.makedirs("results/figures/07_per_day_plots", exist_ok=True)
        for start_day in all_days:
            # Loop
            chosen_dates = pd.date_range(start_day, periods=2).strftime("%Y-%m-%d")
            sub_plot_data = plot_data.copy()
            sub_plot_data = sub_plot_data[plot_data_timesteps.isin(chosen_dates)]
            # plot 1: simple shap values
            sns.set(rc={'figure.figsize': (23, 9)})
            sns.set_style("whitegrid")
            sns.set_context("notebook", font_scale=1)
            plt.figure()
            plot = sub_plot_data.iloc[:, 0:2].plot(linestyle='-',
                                                   use_index=False, linewidth=3, grid=False)
            sub_plot_data.iloc[:, 2:11].plot(kind="bar", stacked=True, width=0.9,
                                           use_index=False, ax=plot, grid=False)
            plot.set(xticklabels=list(pd.to_datetime(sub_plot_data.index).hour))
            plot.legend(title='', loc='upper right',
                        labels=["Predicted residuals", 'Model residuals'] + model_variables_names)
            plot.set_ylim(y_min, y_max)
            ax = plt.gca()
            temp = ax.xaxis.get_ticklabels()
            temp = list(set(temp) - set(temp[::4]))
            for label in temp:
                label.set_visible(False)
            plt.margins(0)
            ax.set(ylabel='Resiudals/SHAP values')
            ax.set(xlabel='Hour')
            os.makedirs("results/figures/07_per_day_plots/" + start_day, exist_ok=True)
            plt.savefig("results/figures/07_per_day_plots/" + start_day + "/07_influence_" + start_day + '.png')
            plt.close()

            # plot 2: shap values with clusters
            sns.set(rc={'figure.figsize': (23, 9)})
            sns.set_style("whitegrid")
            sns.set_context("notebook", font_scale=1)
            plt.figure()
            ax0 = sub_plot_data.iloc[:, 0:2].plot(linestyle='-',
                                                  use_index=False, linewidth=3, grid=False, legend=None)
            ax1 = sub_plot_data.iloc[:, 11:].plot(kind="bar", stacked=True, width=0, sharex=True,
                                                  use_index=False, alpha=0.3, grid=False, legend=None, ax=ax0, cmap="Set1")
            ax2 = sub_plot_data.iloc[:, 11:].plot(kind="bar", stacked=True, width=0.9, sharex=True,
                                                  use_index=False, secondary_y=True, alpha=0.2, grid=False, legend=None,
                                                  ax=ax0, cmap="Set1")
            ax3 = sub_plot_data.iloc[:, 2:11].plot(kind="bar", stacked=True, width=0.9, sharex=True,
                                                   use_index=False, alpha=1, grid=False, legend=None, ax=ax0)
            ax0.set(ylabel='Resiudals/SHAP values')
            ax0.set(xlabel='Hour')
            pal = ["#9b59b6", "#e74c3c", "#34495e", "#2ecc71"]
            empty_ticks = ["" for x in sub_plot_data.index]
            empty_y_ticks = ["" for x in sub_plot_data.index]
            ax1.set(xticklabels=empty_ticks)
            ax2.set(xticklabels=empty_ticks)
            ax2.set(yticklabels=["" for y in ax2.yaxis.get_ticklabels()])
            ax2.tick_params(axis='y', which='both', length=0)
            ax3.set(xticklabels=empty_ticks)
            ax0.set(xticklabels=list(pd.to_datetime(sub_plot_data.index).hour))
            lg = ax3.legend(title='',
                            labels=["Predicted residuals", "Model residuals"] +
                                   ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"] +
                                   model_variables_names,
                            bbox_to_anchor=(1.05, 1.0), loc='upper left', fancybox=True, shadow=True)
            ax1.set_ylim(y_min, y_max)
            plt.title("Residuals, shap values and clusters for two days starting on " + start_day, fontsize=20)
            temp = ax0.xaxis.get_ticklabels()
            temp = list(set(temp) - set(temp[::4]))
            for label in temp:
                label.set_visible(False)
            plt.margins(0)
            # ax.set(ylabel='SHAP values')
            plt.savefig("results/figures/07_per_day_plots/" + start_day + "/07_clustered_influence_" + start_day + '.png',
                        dpi=300,
                        format='png',
                        bbox_extra_artists=(lg,),
                        bbox_inches='tight'
                        )
            plt.close('all')
            # meteo var plots
            for v, variable in enumerate(self.model_variables[2:]):
                variable_data = self.data[pd.to_datetime(self.data.index).strftime("%Y-%m-%d").isin(chosen_dates)]
                plt.figure(figsize=(15, 3))
                sns.set_style("whitegrid", {'axes.grid': False})
                sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2})
                var_plot = sns.lineplot(x=variable_data.index, y=variable, data=variable_data, ci=None)
                plt.margins(0)
                var_plot.set(xticklabels=list(pd.to_datetime(sub_plot_data.index).hour))
                ax = plt.gca()
                temp = ax.xaxis.get_ticklabels()
                temp = list(set(temp) - set(temp[::4]))
                for label in temp:
                    label.set_visible(False)
                plt.savefig("results/figures/07_per_day_plots/" + start_day + "/" + model_variables_names[1:][v] + ".png",
                            dpi=300)
                plt.close()

            plot_imgs = glob.glob("results/figures/07_per_day_plots/" + start_day + "/*.png")
            # plot all pngs into one figure
            fig = plt.figure(figsize=(15, 30))
            columns = 1
            rows = 9
            index = 0
            for i, path in enumerate(plot_imgs):
                if (i == 1):
                    continue
                else:
                    index += 1
                    im = img.imread(path)
                    ax = fig.add_subplot(rows, columns, index)
                    plt.imshow(im, cmap=plt.get_cmap("bone"))
                    ax.grid(False)
                    ax.tick_params(axis='x', colors=(0, 0, 0, 0))
                    ax.tick_params(axis='y', colors=(0, 0, 0, 0))
                    ax.axis("off")
            plt.savefig("results/figures/07_per_day_plots/" + start_day + ".png", dpi = 800)




