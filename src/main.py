# Learning from Mistakes
# Moritz Feigl & Ben Roesky
# Sept 2020

# 1. Directory and imports ---------------------------------------------------------------------------------------------
import os
if os.getcwd() != 'C:/Users/morit/Dropbox/Projekte/Projekt mit Ben/Learning-from-mistakes':
  os.chdir('C:/Users/morit/Dropbox/Projekte/Projekt mit Ben/Learning-from-mistakes')
from load_preprocess_data import load_preprocess_data
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import statsmodels
# 2. Load and preprocess data ------------------------------------------------------------------------------------------
data, x_train, y_train, x_val, y_val, x, y = load_preprocess_data(days_for_validation=5,
                                                                  random_validation=False,
                                                                  seed=42)
model_variables = ['sin_hour', 'cos_hour',
                           'Shortwave Radiation (W/m2)',
                           'Air Temperature (deg C)',
                           'Relative Humidity (%)',
                           'Wind Speed (m/s)',
                           'Longwave radiation',
                           'Precip (mm)',
                           'Discharge (m3/s)',
                           'Tree Temp']
# 3. Explorative Data Analysis -----------------------------------------------------------------------------------------

# 3.1 Residual Histograms
plt.subplots(2, 5)
sns.set(rc={'figure.figsize':(45, 1)})
plot_data = data[['residuals_point_0', 'residuals_point_45',
       'residuals_point_196', 'residuals_point_223', 'residuals_point_324',
       'residuals_point_356', 'residuals_point_494', 'residuals_point_640',
       'residuals_point_771', 'residuals_point_805']]
for i, col in enumerate(plot_data.columns):
    plt.subplot(2, 5, i+1)
    sns.kdeplot(plot_data[col], shade=True, legend = False)
    plt.ylabel('model residuals')
    plt.xlim(-2, 2)
    plt.title(str(plot_data.columns[i]))
#plt.subplots_adjust(right=1.5, top=0.9)
plt.savefig('results/figures/01_residual_hist.png')
plt.clf()

# 3.2 Residual Correlations
corr = data[['residuals_point_0', 'residuals_point_45',
       'residuals_point_196', 'residuals_point_223', 'residuals_point_324',
       'residuals_point_356', 'residuals_point_494', 'residuals_point_640',
       'residuals_point_771', 'residuals_point_805']].corr().round(2)
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 12))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},
            annot=True)
plt.savefig('results/figures/02_residual_corr.png')
plt.clf()
# 3.3 Autocorrelation of point 640 residuals
sns.set(rc={'figure.figsize':(30, 20)})
plot = pd.plotting.autocorrelation_plot(data['residuals_point_640'])
plot.figure.savefig('results/figures/03_residual_autocorr.png')

# 3.3 water temperature time series
sns.set(rc={'figure.figsize':(24, 7)})
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5)
plot_data = data.loc[data["Month"] == 6]
plot_data = plot_data[['wt_predicted_point_640', 'wt_observed_point_640', 'wt_shading_predicted_point_640']]
plot = plot_data.plot(linewidth=2, legend = False)
plot.legend(title='', loc='upper right', labels=['Predicted', 'Observed', 'Shading'])
rmse = np.sqrt(mean_squared_error(plot_data['wt_predicted_point_640'], plot_data['wt_observed_point_640']))
shading_rmse = np.sqrt(mean_squared_error(plot_data['wt_shading_predicted_point_640'], plot_data['wt_observed_point_640']))

plot.set_title(f'Predicted and observed stream water temperature, RMSE = {rmse:.2}, RMSE shading = {shading_rmse:.2}')
plot.figure.savefig('results/figures/04_wt_timeseries_June.png')

sns.set(rc={'figure.figsize':(24, 7)})
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5)
plot_data = data.loc[data["Month"] == 8]
plot_data = plot_data[['wt_predicted_point_640', 'wt_observed_point_640', 'wt_shading_predicted_point_640']]
plot = plot_data.plot(linewidth=2, legend = False)
plot.legend(title='', loc='upper right', labels=['Predicted', 'Observed', 'Shading'])
rmse = np.sqrt(mean_squared_error(plot_data['wt_predicted_point_640'], plot_data['wt_observed_point_640']))
shading_rmse = np.sqrt(mean_squared_error(plot_data['wt_shading_predicted_point_640'], plot_data['wt_observed_point_640']))

plot.set_title(f'Predicted and observed stream water temperature, RMSE = {rmse:.2}, RMSE shading = {shading_rmse:.2}')
plot.figure.savefig('results/figures/04_wt_timeseries_August.png')

# 3.4 Feature correlation
corr = data[model_variables].corr().round(2)
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))
# Set up the matplotlib figure
sns.set_context("notebook", font_scale=1)
f, ax = plt.subplots(figsize=(15, 14))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},
            annot=True)
plt.savefig('results/figures/05_feature_correlation.png')