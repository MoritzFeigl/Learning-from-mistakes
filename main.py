# Learning from Mistakes
# Moritz Feigl & Ben Roesky
# Sept 2020

# 1. Directory and imports
import os

if os.getcwd() != 'C:/Users/morit/Dropbox/Projekte/Projekt mit Ben/Learning-from-mistakes':
    os.chdir('C:/Users/morit/Dropbox/Projekte/Projekt mit Ben/Learning-from-mistakes')
import src.utils as utils
import src.models as models

# 2. Load and preprocess data
data, x_train, y_train, x_val, y_val, x, y = utils.load_preprocess_data(days_for_validation=5,
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

# 3. Explorative Data Analysis
utils.residual_plots(data)
utils.eda_plots(data, model_variables, '640')

# 4. Variable Influence with regression model
regression_model = models.RegressionModel(x, y)
regression_model.variance_inflation()
regression_model.center_data()
regression_model.variance_inflation()
regression_model.fit()

# 5. Residual prediction model
residual_model = models.XGBoost(x_train, y_train, x_val, y_val, x, y, data)
lag_variables = ['Shortwave Radiation (W/m2)',
                 'Air Temperature (deg C)',
                 'Relative Humidity (%)',
                 'Wind Speed (m/s)',
                 'Longwave radiation',
                 'Precip (mm)',
                 'Discharge (m3/s)',
                 'Tree Temp']
residual_model.create_lagged_features(lag_variables, lag=4)
print(residual_model.x_train.columns)
# residual_model.hyperpar_optimization(init_points=20, n_iter=80, n_jobs=10,
#                                     model_run_name="residual_xgboost_lag4")
residual_model.fit(model_run_name="residual_xgboost_lag4")
residual_model.plot_prediction(model_run_name="residual_xgboost_lag4")

# TODO: add shap plots to residual model class -> aggregate shap values of lagged features
import shap

explainer = shap.TreeExplainer(residual_model.model)
x = create_lags(x, lag_variables, 4)
y = y.iloc[4:]
shap_values = explainer.shap_values(x)
shap.summary_plot(shap_values, x)
shap.dependence_plot('Tree Temp', shap_values, x)

prediction = pd.DataFrame({"prediction": residual_model.model.predict(x)})
prediction.index = x.index

# prepare shap data
shap_df = pd.DataFrame(shap_values)
shap_df.columns = x.columns
shap_df.index = residual_model.data.index
# for var in model_variables[1:]:
#  shap_df.loc[data[var] == 0, var] = 0
xgb_prediction = prediction["prediction"]
xgb_prediction.index = residual_model.data.index
plot_data = pd.concat([xgb_prediction, y, shap_df], axis=1)
# plot limits for y-axis
y_max = np.max([y.iloc[:, 0], xgb_prediction.values])
y_min = np.min([y.iloc[:, 0], xgb_prediction.values])
# plot in 2 day steps
all_days = pd.to_datetime(data.index).strftime("%Y-%m-%d").unique()[::2]

for start_day in all_days:
    chosen_dates = pd.date_range(start_day, periods=2).strftime("%Y-%m-%d")
    plot_data.index = pd.to_datetime(plot_data.index).strftime('%d-%m-%Y')
    plot_data = plot_data[pd.to_datetime(plot_data.index).isin(chosen_dates)]

    # plot 1
    sns.set(rc={'figure.figsize': (23, 9)})
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1)
    plt.figure()
    # plot = pl_data.iloc[:,2:].plot(kind="bar", stacked=True, width = 0.9,
    #                                  use_index=True)
    plot = plot_data.iloc[:, 0:2].plot(linestyle='-',  # ax = plot,
                                       use_index=False, linewidth=3)
    plot_data.iloc[:, 2:].plot(kind="bar", stacked=True, width=0.9,
                               use_index=False, ax=plot)
    plot.set(xticklabels=list(pd.to_datetime(plot_data.index).hour))
    plot.legend(title='', loc='upper right',
                labels=["Predicted residuals", 'Model residuals'] + list(x.columns))
    plot.set_ylim(y_min, y_max)
    ax = plt.gca()
    temp = ax.xaxis.get_ticklabels()
    temp = list(set(temp) - set(temp[::4]))
    for label in temp:
        label.set_visible(False)
    plt.margins(0)
    ax.set(ylabel='SHAP values')
    plt.savefig('results/figures/06_influence_' + start_day +'.png')

    # Plot 2
    variable_data = data[pd.to_datetime(data.index).strftime("%Y-%m-%d").isin(chosen_dates)]
    plt.figure(figsize=(23, 4))
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2})
    sns.lineplot(x=variable_data.index, y=variable, data=variable_data, ci=None)
    plt.margins(0)
    plt.show()
