# Learning from Mistakes
# Moritz Feigl
# July 2021

import pandas as pd
import numpy as np
from src.utils import load_preprocess_data, residual_plots, data_plots, create_pca_features
import src.models as models

# 1. Load and preprocess data
model_variables = ['sin_hour', 'cos_hour',
                   'Shortwave Radiation (W/m2)',
                   'Air Temperature (deg C)',
                   'Relative Humidity (%)',
                   'Wind Speed (m/s)',
                   'Longwave radiation',
                   'Discharge (m3/s)',
                   'Tree Temp']
lag_variables = model_variables[2:]
lag = 8
data, x_train, y_train, x_test, y_test, x, y = load_preprocess_data(days_for_validation=5,
                                                                    lag_variables=lag_variables,
                                                                    random_validation=True,
                                                                    seed=42, lag=lag, reload=True, save_csv=True)

# 2. correlation between stream temperature measurement points
res_data = data[['residuals_point_45',
                 'residuals_point_196', 'residuals_point_223', 'residuals_point_324',
                 'residuals_point_356', 'residuals_point_494', 'residuals_point_640',
                 'residuals_point_771', 'residuals_point_805']]
res_data.corr().mean().mean()
cors = res_data.corr().where(np.triu(np.ones(res_data.corr().shape), k=1).astype(bool)).stack().reset_index()
cors.median()

# 3. Descriptive plots (including prediction for hypotheses tests)
residual_plots(data)
data_plots(data, model_variables, '640')

# 4. Variable Influence with regression model
regression_model = models.RegressionModel(x, y, model_variables)
regression_model.variance_inflation()
regression_model.center_data()
regression_model.variance_inflation()
regression_model.fit()

# 5. PCA features
x_pca, x_pca_train, x_pca_test, loadings = create_pca_features(x, x_train, x_test)

# 6. Residual prediction model
residual_model = models.XGBoost(x_pca_train, y_train, x_pca_test, y_test, x_pca, y, data, model_variables)
model_name = f"xgboost_pca_lag{lag}_rand5Val_noPrecip"
overwrite = False
residual_model.hyperpar_optimization(init_points=10, n_iter=40, n_jobs=16,
                                     model_run_name=model_name, overwrite=overwrite)
residual_model.fit(model_run_name=model_name)
residual_model.plot_prediction(model_run_name=model_name)
if overwrite:
    with open('results/tables/lag_search.txt', 'a') as f:
        f.write(f'Lag: {lag}: {residual_model.test_rmse} test RMSE\n')
residual_model.print_rmse()

# 7. SHAP values
residual_model.compute_shap_values(loadings)

# 8. Clustering and cluster plots
residual_model.cluster_shap_values(chosen_algorithm="kmean", chosen_n_cluster=3, max_clusters=3, kmeans_seed=42)
residual_model.plot_variable_importance()
residual_model.plot_cluster_properties()

# 9. Cluster specific properties
# table of cluster variable values
cluster_data = residual_model.cluster_data.copy()
cluster_data["Residuals"] = y
cluster_data["Cluster"] = residual_model.cluster_df
cluster_data["temp_diff"] = cluster_data["Tree Temp"] - cluster_data["Air Temperature (deg C)"]
cluster_data.to_excel("results/tables/cluster_data.xlsx")
median_cluster_values = cluster_data.groupby("Cluster").median().round(2).transpose()
median_cluster_values.columns = [f"Cluster {str(x + 1)}" for x in median_cluster_values.columns]
median_cluster_values.to_excel("results/tables/median_cluster_values.xlsx")

# table of median cluster shap values
shap_vals = pd.DataFrame(residual_model.aggregated_shap_values)
shap_vals["Cluster"] = residual_model.cluster_df
shap_vals.to_excel("results/tables/cluster_shap_data.xlsx")
median_cluster_shap = shap_vals.groupby('Cluster').median().round(2).transpose()
median_cluster_shap.columns = [f"Cluster {str(x + 1)}" for x in median_cluster_shap.columns]
median_cluster_shap.to_excel("results/tables/median_shap_values.xlsx")
