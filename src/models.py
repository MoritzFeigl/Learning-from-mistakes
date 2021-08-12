import statsmodels.stats.outliers_influence as out
import statsmodels.api as api
import statsmodels.tools as tools
import sklearn.metrics as metrics
import sklearn.model_selection as model_selection
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from math import pi
import re
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
import sklearn.preprocessing as preprocessing


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
        data_centered = pd.DataFrame(preprocessing.scale(self.x, with_mean='True', with_std='False'))
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
    rkf = model_selection.RepeatedKFold(n_splits=5, n_repeats=5, random_state=1234)

    cval = model_selection.cross_val_score(estimator, data, targets, cv=rkf,
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

    def __init__(self, x_train, y_train, x_test, y_test, x, y, data, model_variables):
        self.model_variables = model_variables
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x = x
        self.y = y
        self.data = data

    def hyperpar_optimization(self, init_points, n_iter, n_jobs, model_run_name, overwrite):
        """ Applies bayesian hyperparameter optimization for XGBoost parameters """
        if os.path.isfile("models/" + model_run_name + "_optimized_parameter.txt") and not overwrite:
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
        prediction = self.model.predict(self.x_test)
        self.test_rmse = np.sqrt(metrics.mean_squared_error(self.y_test, prediction))
        print("Optimized model RMSE in test set: %f" % self.test_rmse)

        # feature importance
        feature_important = self.model.get_booster().get_score(importance_type='gain')
        keys = list(feature_important.keys())
        values = list(feature_important.values())
        feature_data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=True)
        sns.set(rc={'figure.figsize': (17, 6)})
        feature_data.plot(kind='barh', title="XGBoost feature importance")
        # plt.show()
        print("Saved XGBoost feature importance in results/figures/06_xgboost_feature_importance")
        plt.savefig("results/figures/06_xgboost_feature_importance")

    def print_rmse(self):
        test_prediction = self.model.predict(self.x_test)
        self.test_rmse = round(np.sqrt(metrics.mean_squared_error(self.y_test, test_prediction)), 4)
        train_prediction = self.model.predict(self.x_train)
        self.train_rmse = round(np.sqrt(metrics.mean_squared_error(self.y_train, train_prediction)), 4)
        full_prediction = self.model.predict(self.x)
        self.full_rmse = round(np.sqrt(metrics.mean_squared_error(self.y, full_prediction)), 4)

        print(f"full RMSE:{self.full_rmse}, train RMSE: {self.train_rmse}, test RMSE: {self.test_rmse}")
    def plot_prediction(self, model_run_name):
        prediction = pd.DataFrame({"prediction": self.model.predict(self.x)})
        prediction.index = self.x.index

        xbgoost_predicted_wt = self.data['wt_predicted_point_640'] - prediction["prediction"]
        plot_data = pd.concat(
            [self.data[['wt_observed_point_640', 'wt_predicted_point_640', 'calibration_validation']],
             xbgoost_predicted_wt], axis=1)
        plot_data.index = pd.to_datetime(plot_data.index)
        sns.set(rc={'figure.figsize': (24, 7)})
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.5)
        plot = plot_data.plot(linewidth=2, legend=False)
        plot.legend(title='', loc='upper right', fontsize=18,
                    labels=['Observation', 'HFLUX Prediction', 'HFLUX + Error Model'])
        plot.set_ylabel('Stream Temperature (°C)', fontsize=24)
        plot.tick_params(axis='x', labelsize=18)
        plot.tick_params(axis='y', labelsize=18)
        plot.set_xlabel('', fontsize=24)
        plot.figure.savefig('results/figures/06_XGBoost_prediction_' + model_run_name + '.png',
                            bbox_inches='tight', dpi=600)
        plt.close("all")

    def compute_shap_values(self, loadings):
        explainer = shap.Explainer(self.model)
        shap_array = explainer.shap_values(self.x)
        # aggregate lagged shap values
        self.shap_values = pd.DataFrame(shap_array)
        self.shap_values.columns = self.x.columns
        self.shap_values.index = self.data.index

        # turn PCA shap into feature shap with loadings
        aggregated_shap_loadings = pd.DataFrame(0, index=self.shap_values.index, columns=loadings.index)
        for pca_col in self.shap_values.columns:
            col_shap = self.shap_values[pca_col]
            col_loading = loadings[pca_col]
            shap_loadings = []
            for i, loading in enumerate(col_loading):
                shap_loadings.append(col_shap * loading)
            shap_loadings = pd.concat(shap_loadings, axis=1)
            shap_loadings.columns = col_loading.index
            aggregated_shap_loadings = aggregated_shap_loadings.add(shap_loadings, fill_value=0)

        var_names = list(map(lambda v: re.split('_', v)[0], self.model_variables))
        lag_variables = [x for x in var_names if x not in ["sin", "cos"]]
        self.aggregated_shap_values = pd.DataFrame(
            {"hour": aggregated_shap_loadings[["sin_hour", "cos_hour"]].sum(axis=1)})
        self.model_variables_clean = list(map(lambda v: re.split(' \(', v)[0], lag_variables))
        for variable in self.model_variables_clean:
            # aggregate shap values over lags
            self.aggregated_shap_values[variable] = aggregated_shap_loadings.filter(
                regex=(variable + "*")).sum(axis=1)


    def plot_shap_dendrogram(self):
        dend = shc.dendrogram(shc.linkage(self.shap_values, method='complete'))
        plt.savefig("results/figures/07_clustering/dendrogram_complete")
        plt.close("all")
        dend_ = shc.dendrogram(shc.linkage(self.shap_values, method='ward'))
        plt.savefig("results/figures/07_clustering/dendrogram_ward")
        plt.close("all")

    def cluster_shap_values(self, chosen_algorithm, chosen_n_cluster, max_clusters=10, min_clusters=3,
                            kmeans_seed=10):
        range_n_clusters = range(min_clusters, max_clusters + 1)
        silhouette_scores = {"kmean": dict(),
                             "hierarchical_ward": dict(),
                             "hierarchical_complete": dict()}
        print("Cluster results for number of clusters in 3-10 are saved under results/figures/07_clusterinf")
        for n_clusters in range_n_clusters:
            cluster_algorithms = [["kmean", KMeans(n_clusters=n_clusters, random_state=kmeans_seed, n_init=20)],
                                  ["hierarchical_ward", AgglomerativeClustering(n_clusters=n_clusters)],
                                  ["hierarchical_complete",
                                   AgglomerativeClustering(n_clusters=n_clusters, linkage="complete")]]
            for cluster_alg in cluster_algorithms:

                # Create a subplot with 1 row and 2 columns
                fig, (ax1, ax2) = plt.subplots(1, 2)
                fig.set_size_inches(18, 7)
                # The 1st subplot is the silhouette plot
                ax1.set_xlim([-0.1, 1])
                ax1.set_ylim([0, len(self.shap_values) + (n_clusters + 1) * 10])
                # clustering
                clusterer = cluster_alg[1]  # KMeans(n_clusters=n_clusters, random_state=10)
                cluster_labels = clusterer.fit_predict(self.shap_values)
                # The silhouette_score gives the average value for all the samples.
                # This gives a perspective into the density and separation of the formed
                # clusters
                silhouette_avg = silhouette_score(self.shap_values, cluster_labels)
                print("For n_clusters =", n_clusters, cluster_alg[0],
                      "The average silhouette_score is :", round(silhouette_avg, 3))
                silhouette_scores[cluster_alg[0]][n_clusters] = silhouette_avg
                # Compute the silhouette scores for each sample
                sample_silhouette_values = silhouette_samples(self.shap_values, cluster_labels)

                # pca for visualization
                pca = PCA(n_components=2, svd_solver='full')
                pca.fit(self.shap_values)
                pca_X = pca.transform(self.shap_values)
                y_lower = 10
                # color map
                colors = plt.cm.get_cmap("Set1", n_clusters)

                for i in range(n_clusters):
                    # Aggregate the silhouette scores for samples belonging to
                    # cluster i, and sort them
                    ith_cluster_silhouette_values = \
                        sample_silhouette_values[cluster_labels == i]

                    ith_cluster_silhouette_values.sort()

                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i

                    color = tuple(colors.colors[i])
                    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                      0, ith_cluster_silhouette_values,
                                      facecolor=color, edgecolor=color, alpha=0.7)

                    # Label the silhouette plots with their cluster numbers at the middle
                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i+1))

                    # Compute the new y_lower for next plot
                    y_lower = y_upper + 10  # 10 for the 0 samples

                ax1.set_title("Silhouette plot")
                ax1.set_xlabel("Silhouette coefficient values")
                ax1.set_ylabel("Cluster")

                # The vertical line for average silhouette score of all the values
                ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

                ax1.set_yticks([])  # Clear the yaxis labels / ticks
                ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                # 2nd Plot showing the actual clusters formed
                color_array = np.array(list(map(tuple, colors.colors)))
                point_col_array = np.empty(shape=(cluster_labels.shape[0], 4))
                for row in range(cluster_labels.shape[0]):
                    point_col_array[row, :] = color_array[cluster_labels[row]]
                ax2.scatter(pca_X[:, 0], pca_X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                            c=point_col_array, edgecolor='k')

                ax2.set_title("Clusters shown in Principal components of data")
                ax2.set_xlabel(f"1st Principal Component ({100*pca.explained_variance_ratio_[0]:.3}%)")
                ax2.set_ylabel(f"2st Principal Component ({100*pca.explained_variance_ratio_[1]:.3}%)")

                plt.suptitle(f"Silhouette analysis for {cluster_alg[0]} clustering with {n_clusters} clusters",
                             fontsize=14, fontweight='bold')
                os.makedirs("results/figures/07_clustering", exist_ok=True)
                plt.savefig(f"results/figures/07_clustering/cluster_shillouette_{cluster_alg[0]}_ncluster_{n_clusters}",
                            dpi=600, bbox_inches='tight')
                plt.close()

        cluster_algorithms_ = [KMeans(n_clusters=chosen_n_cluster, random_state=kmeans_seed),
                               AgglomerativeClustering(n_clusters=chosen_n_cluster),
                               AgglomerativeClustering(n_clusters=chosen_n_cluster,
                                                       linkage="complete")]

        cluster_algorithms_names = ["kmean", "hierarchical_ward", "hierarchical_complete"]
        cluster_alg_id = cluster_algorithms_names.index(chosen_algorithm)
        self.optimal_n_clusters = chosen_n_cluster
        self.cluster_model = cluster_algorithms_[cluster_alg_id]
        self.cluster_labels = self.cluster_model.fit_predict(self.shap_values)
        print(f"Estimated clusters with {chosen_algorithm} with {self.optimal_n_clusters} clusters.")

    def plot_variable_importance(self):
        # define Cluster dummy columns
        dummies = pd.get_dummies(self.cluster_labels)
        dummies.index = self.y.index

        # get prediction
        prediction = pd.DataFrame({"prediction": self.model.predict(self.x)})
        prediction.index = self.data.index

        # define df for plotting
        plot_data = pd.concat(
            [prediction["prediction"], self.y, self.aggregated_shap_values, dummies], axis=1)

        plot_data.index = self.data.index
        plot_data.to_csv("results/tables/dat_and_shap.csv")
        plot_data_timesteps = pd.to_datetime(self.data.index).strftime("%Y-%m-%d")
        # plot limits for y-axis
        shaps = self.aggregated_shap_values.copy()
        shaps[shaps < 0] = 0
        shap_max = shaps.sum(axis=1).max()
        shaps = self.aggregated_shap_values.copy()
        shaps[shaps > 0] = 0
        shap_min = shaps.sum(axis=1).min()

        y_max = np.max([self.y.values.max(), prediction["prediction"].max(), shap_max])
        y_min = np.min([self.y.values.min(), prediction["prediction"].min(), shap_min])

        data_tmp = self.data.copy()
        if("Wind Speed (m/s)" in self.data.columns):
            data_tmp.loc[data_tmp["Wind Speed (m/s)"] > 2.6, "Wind Speed (m/s)"] = np.nan
        var_max = data_tmp[self.model_variables[2:]].max()
        var_min = data_tmp[self.model_variables[2:]].min()

        # plot in 2 day steps
        all_days = pd.to_datetime(self.data.index).strftime("%Y-%m-%d").unique()[::2]
        os.makedirs("results/figures/08_per_day_plots", exist_ok=True)
        index_of_last_var = self.aggregated_shap_values.shape[1] + 2
        var_names = list(self.aggregated_shap_values.columns)
        for i, vs in enumerate(var_names):
            if vs == "Tree Temp":
                var_names[i] = "Canopy Temperature"
            elif vs == "hour":
                var_names[i] = "Hour"
            elif vs == "Longwave radiation":
                var_names[i] = "Longwave Radiation"

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
            plot = sub_plot_data.iloc[:, 2:index_of_last_var].plot(kind="bar", stacked=True, width=0.9,
                                                                   use_index=False, grid=False)
            ax0 = sub_plot_data.iloc[:, 0:2].plot(linestyle='-', color=["black", "#636161"],
                                                  use_index=False, linewidth=3, grid=False, ax=plot)
            ax0.set_ylabel('Residuals/PCA SHAP values', fontsize=24)
            ax0.set_xlabel('Hour', fontsize=24)
            ax0.set_xticklabels(list(pd.to_datetime(sub_plot_data.index).hour), {"fontsize": 18})
            ax0.tick_params(axis='y', labelsize=18)
            lg = ax0.legend(title='',
                            labels=["Predicted Residuals", 'HFLUX Residuals'] + var_names, fontsize=18,
                            bbox_to_anchor=(1.01, 1.0), loc='upper left', fancybox=True, shadow=True)
            ax0.set_ylim(y_min, y_max)
            ax = plt.gca()
            temp = ax.xaxis.get_ticklabels()
            temp = list(set(temp) - set(temp[::4]))
            for label in temp:
                label.set_visible(False)
            plt.xticks(rotation=0)
            plt.margins(0)
            os.makedirs("results/figures/08_per_day_plots/" + start_day, exist_ok=True)
            plt.savefig("results/figures/08_per_day_plots/" + start_day + "/07_influence_" + start_day + '.png',
                        dpi=600, format='png', bbox_extra_artists=(lg,), bbox_inches='tight')
            plt.close("all")

            # plot 2: shap values with clusters
            sns.set(rc={'figure.figsize': (23, 9)})
            sns.set_style("whitegrid")
            sns.set_context("notebook", font_scale=1)
            plt.figure()
            ax1 = sub_plot_data.iloc[:, index_of_last_var:].plot(kind="bar", stacked=True, width=0, sharex=True,
                                                                 use_index=False, alpha=0.3, grid=False, legend=None,
                                                                 cmap="Set1")
            my_palette = plt.cm.get_cmap("Set1", dummies.shape[1])
            pos_clusters = y_max * sub_plot_data.iloc[:, index_of_last_var:]
            ax2 = pos_clusters.plot(kind="bar", stacked=True, width=0.9, sharex=True,
                                    use_index=False, alpha=0.2,
                                    grid=False, legend=None,
                                    ax=ax1, cmap=my_palette)
            neg_clusters = y_min * sub_plot_data.iloc[:, index_of_last_var:]
            ax2_neg = neg_clusters.plot(kind="bar", stacked=True, width=0.9, sharex=True,
                                        use_index=False, alpha=0.2,
                                        grid=False, legend=None,
                                        ax=ax2, cmap=my_palette)
            ax3 = sub_plot_data.iloc[:, 2:index_of_last_var].plot(kind="bar", stacked=True, width=0.9, sharex=True,
                                                                  use_index=False, alpha=1, grid=False, legend=None,
                                                                  ax=ax2_neg)
            ax0 = sub_plot_data.iloc[:, 0:2].plot(linestyle='-', ax=ax3, color=["black", "#636161"],
                                                  use_index=False, linewidth=3, grid=False, legend=None)
            ax0.set_ylabel('Residuals/PCA SHAP values', fontsize=24)
            ax0.set_xlabel('Hour', fontsize=24)
            empty_ticks = ["" for x in sub_plot_data.index]
            ax1.set(xticklabels=empty_ticks)
            ax2.set(xticklabels=empty_ticks)
            ax3.set(xticklabels=empty_ticks)
            ax0.set_xticklabels(list(pd.to_datetime(sub_plot_data.index).hour), {"fontsize": 18})
            ax0.tick_params(axis='y', labelsize=18)
            handles, labels = ax0.get_legend_handles_labels()
            new_handles = [handles[i] for i in range(len(handles)) if i not in range(2, 8)]
            new_labels = ["Predicted Residuals", "HFLUX Residuals"] + \
                         ["Cluster " + str(x + 1) for x in range(dummies.shape[1])] + \
                         var_names
            lg = ax0.legend(new_handles, new_labels, fontsize=18,
                       bbox_to_anchor=(1.01, 1.0), loc='upper left', fancybox=True, shadow=True)
            ax1.set_ylim(y_min, y_max)
            temp = ax0.xaxis.get_ticklabels()
            temp = list(set(temp) - set(temp[::4]))
            for label in temp:
                label.set_visible(False)
            plt.xticks(rotation=0)
            plt.margins(0)
            plt.savefig("results/figures/08_per_day_plots/" + start_day + "/07_clustered_influence_" + start_day + '.png',
                dpi=600, format='png', bbox_extra_artists=(lg,), bbox_inches='tight')
            plt.savefig("results/figures/08_per_day_plots/07_clustered_influence_" + start_day + '.png',
                dpi=600, format='png', bbox_extra_artists=(lg,), bbox_inches='tight')
            plt.close('all')
            # meteo var plots
            for v, variable in enumerate(self.model_variables[2:]):
                variable_data = self.data[
                    pd.to_datetime(self.data.index).strftime("%Y-%m-%d").isin(chosen_dates)]
                plt.figure(figsize=(15, 3))
                sns.set_style("whitegrid", {'axes.grid': False})
                sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2})
                var_plot = sns.lineplot(data=variable_data, x=variable_data.index, y=variable, ci=None)
                ticks = variable_data.index[variable_data.Minute == 0]
                var_plot.set_xticks(ticks)
                var_plot.set_xticklabels(variable_data.Hour[::4])
                plt.margins(0)
                if variable == 'Air Temperature (deg C)':
                    var_plot.set_ylim(var_min['Tree Temp'], var_max['Tree Temp'])
                else:
                    var_plot.set_ylim(var_min[variable], var_max[variable])
                plt.savefig("results/figures/08_per_day_plots/" + start_day + "/" +
                            re.sub("/", "_", self.model_variables[2:][v]) + ".png",
                            dpi=600, bbox_inches='tight')
                plt.close()

    def plot_cluster_properties(self):
        print("Saving Cluster propeties figures in results/figures/09_Cluster_properties")
        # Cluster properties
        shap_columns = list(self.aggregated_shap_values.columns)
        self.cluster_df = pd.DataFrame({"Cluster": self.cluster_labels})
        self.cluster_df.index = self.x.index
        self.cluster_data = pd.concat([pd.DataFrame({"Residuals": self.y.abs().squeeze()}),
                                       self.data.Hour,
                                       self.data.loc[:, self.model_variables[2:]]], axis=1)
        if ("Wind Speed (m/s)" in self.cluster_data.columns):
            self.cluster_data.loc[self.cluster_data["Wind Speed (m/s)"] > 2.6, "Wind Speed (m/s)"] = np.nan
        # boxplots
        box_scaler = preprocessing.MinMaxScaler()
        transformed_cluster_data = pd.DataFrame(box_scaler.fit_transform(self.cluster_data))
        transformed_cluster_data.columns = ['Residuals'] + shap_columns
        transformed_cluster_data.index = self.cluster_df.index
        transformed_cluster_data["Cluster"] = self.cluster_df
        os.makedirs("results/figures/09_Cluster_properties", exist_ok=True)
        plt.close('all')
        res_col = ["#636161"] + sns.color_palette().as_hex()
        my_pal = {}
        for l in range(len(transformed_cluster_data.columns) - 1):
            my_pal[transformed_cluster_data.columns[l]] = res_col[l]
        for i_cluster in range(self.optimal_n_clusters):
            sns.set(rc={'figure.figsize': (20, 12)})
            sns.set_style("whitegrid")
            single_cluster = transformed_cluster_data[transformed_cluster_data["Cluster"] == i_cluster].drop("Cluster",
                                                                                                             axis=1)
            df_long = pd.melt(single_cluster, var_name="variable", value_name="value")
            ax = sns.boxplot(x=df_long.variable, y=df_long.value, palette=my_pal)
            ax.set_xlabel("", fontsize=0)
            ax.tick_params(labelsize=16)
            ax.set_ylabel("Scaled values", fontsize=25)
            plt.tight_layout()
            plt.savefig(
                "results/figures/09_Cluster_properties/09_Cluster" + str(i_cluster + 1) + "_variable_distribution",
                dpi=600)
            plt.close()

        # Spider plots
        # scale all shap values to [0,1] with global min/max
        scaler = preprocessing.MinMaxScaler()
        shap_vals = self.aggregated_shap_values.abs()
        shap_vals_one_column = shap_vals.to_numpy().reshape([-1, 1])
        result_one_column = scaler.fit_transform(shap_vals_one_column)
        transformed_shap = pd.DataFrame(result_one_column.reshape(shap_vals.shape))
        transformed_shap.columns, transformed_shap.index = shap_vals.columns, shap_vals.index
        # add cluster labels
        cluster_shap = pd.concat([transformed_shap, self.cluster_df], axis=1)
        cluster_shap_data = cluster_shap.groupby('Cluster').median()
        var_names = list(self.aggregated_shap_values.columns)
        for i, vs in enumerate(var_names):
            if vs == "Tree Temp":
                var_names[i] = "Canopy Temperature"
            elif vs == "hour":
                var_names[i] = "Hour"
            elif vs == "Longwave radiation":
                var_names[i] = "Longwave Radiation"
        cluster_shap_data.columns = var_names

        def make_spider(cluster, color):
            max_val = cluster_shap_data.max().max()
            # number of variable
            categories = list(cluster_shap_data)
            n = len(categories)
            # angle of each axis in the plot
            angles = [nn / float(n) * 2 * pi for nn in range(n)]
            angles += angles[:1]
            plt.figure(figsize=(9, 9), dpi=600)
            # Initialise the spider plot
            ax = plt.subplot(1, 1, 1, polar=True)
            # first axis on top:
            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)
            # Draw ylabels
            ax.set_rlabel_position(0)
            ax.set_yticklabels([])
            values = cluster_shap_data.loc[cluster].values.flatten().tolist()
            values += values[:1]
            plt.ylim(0, max_val)  # max(values))
            ax.plot(angles, values, color=color, linewidth=3, linestyle='solid')
            ax.fill(angles, values, color=color, alpha=0.5, zorder=1.9)
            plt.subplots_adjust(left=0.1, right=0.7)
            # Draw one axe per variable + add labels labels yet
            ax.set_xticklabels([])
            ax.grid(False)
            plt.xticks(angles[:-1], categories, color='black', size=18, zorder=2)
            ax.yaxis.set_zorder(0)
            ax.set_rlabel_position(285)  # get radial labels away from plotted line
            # make custom xgrid
            ax.xaxis.set_zorder(3)
            for cg in angles:
                ax.plot((0, cg), (0, max_val), c="silver", zorder=1, linewidth=0.5)
            # make custom ygrid
            rticks = np.arange(0, max_val, max_val / 5)
            x = np.arange(0, 2 * np.pi, 0.01)
            y = np.outer(np.ones(x.shape), rticks)
            ax.plot(x, y, zorder=1, color='silver', linestyle='-', linewidth=0.5)
            # Save figure
            os.makedirs("results/figures/09_Cluster_properties", exist_ok=True)
            plt.savefig("results/figures/09_Cluster_properties/Cluster_" + str(cluster + 1),
                        dpi=600, bbox_inches='tight')
            plt.close()

        # Create a color palette:
        my_palette = plt.cm.get_cmap("Set1", len(cluster_shap_data.index))
        # Loop to plot
        for cluster in range(self.optimal_n_clusters):
            make_spider(cluster, color=my_palette(cluster))

        # cluster counts
        sns.set_style("whitegrid")
        cluster_infos = np.unique(self.cluster_labels, return_counts=True)
        bars = sns.barplot(x=cluster_infos[0], y=cluster_infos[1], palette=my_palette.colors)
        bars.set_ylabel("Counts", fontsize=25)
        bars.set_xlabel("Cluster", fontsize=25)
        bars.tick_params(labelsize=16)
        for i, p in enumerate(bars.patches):
            height = p.get_height()
            bars.text(p.get_x() + p.get_width() / 2., height + 0.1, cluster_infos[1][i], ha="center", fontsize=16)
        plt.savefig("results/figures/09_Cluster_properties/Cluster_counts")
        plt.close("all")
        # plot for pos and neg residuals
        residual_cluster_df = pd.concat([pd.DataFrame({"Residuals": self.y.squeeze()}),
                                         self.cluster_df], axis=1)
        residual_cluster_df["positive"] = residual_cluster_df["Residuals"] > 0
        # boxplots
        for cluster in range(self.optimal_n_clusters):
            ax = sns.countplot(x='positive', data=residual_cluster_df[residual_cluster_df["Cluster"] == cluster],
                               palette=["b", "r"])
            ax.set_xticklabels(["Negative", "Positive"])
            ax.set_xlabel("", fontsize=0)
            plt.title("Residuals", fontsize=40)
            ax.tick_params(labelsize=30)
            ax.set_ylabel("Count", fontsize=30)
            plt.savefig("results/figures/09_Cluster_properties/Cluster_" + str(cluster + 1) + "_residual_signs",
                        dpi=600, bbox_inches='tight')
            plt.close('all')
