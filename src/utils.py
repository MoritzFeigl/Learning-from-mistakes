# Imports
from os.path import isfile
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

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

def load_preprocess_data(days_for_validation, random_validation=False, seed=None, lag=6, overwrite=False):
    """
    Load and preprocess all data sets for the learning from mistakes project.
    The concatenated data set is stored as csv in processed/data.csv and also returned as DataFrame

    Parameters
    ----------
    days_for_validation: int, number of days for validation
    random_validation: True/False should validation set taken randomly or the end of the time series
    seed: int, random seed for random validation set creation

    Returns
    -------
    data: DataFrame containing all preprocessed and concatenated variables
    """
    if isfile('data/processed/data.csv') and not overwrite:
        print('Load previously computed data set from "data/preprocessed/data.csv"')
        data = pd.read_csv('data/processed/data.csv')
        x_train = pd.read_csv("data/processed/x_train.csv")
        y_train = pd.read_csv("data/processed/y_train.csv")
        x_val = pd.read_csv("data/processed/x_val.csv")
        y_val = pd.read_csv("data/processed/y_val.csv")
        x = pd.read_csv("data/processed/x.csv")
        y = pd.read_csv("data/processed/y.csv")
    else:

        append_data = []
        for index in ['C', 'V', 'V3']:
            # Meteorological Data
            met_data = pd.read_excel('data/raw/Input' + index + '.xlsx', sheet_name="met_data")
            precip = pd.read_excel('data/raw/Input' + index + '.xlsx', sheet_name="precip")
            dis_data = pd.read_excel('data/raw/Input' + index + '.xlsx', sheet_name="dis_data", skiprows=1, header=None)
            discharge_805 = pd.DataFrame({'Discharge (m3/s)': dis_data.iloc[4, 1:].transpose()})
            # observed wt
            wt_observed = pd.read_excel('data/raw/Input' + index + '.xlsx', sheet_name="temp", header=None).transpose()
            measurement_points = pd.read_excel('data/raw/Input' + index + '.xlsx', sheet_name="temp_t0_data")
            wt_observed.columns = ["wt_observed_point_" + str(i) for i in measurement_points["Distance (m)"]]
            # observed wt at boundary
            x0_data = pd.read_excel('data/raw/Input' + index + '.xlsx', sheet_name="temp_x0_data")
            x0_data = x0_data.drop(labels='Time (min)', axis=1)
            x0_data.columns = ['x0 Temperature (deg C)']
            # predicted wt
            wt_predicted = pd.read_csv('data/raw/Output' + index + '.csv',
                                       header=None)  # rows: m of stream, columns: timesteps in min
            # get only relevant points and every 15th time steps
            wt_predicted = wt_predicted.iloc[measurement_points["Distance (m)"]]
            wt_predicted = wt_predicted.iloc[:, ::15].transpose()
            wt_predicted.columns = ["wt_predicted_point_" + str(i) for i in measurement_points["Distance (m)"]]
            # get shading predictions
            measurement_points = pd.read_excel('data/raw/Input' + index + '.xlsx', sheet_name="temp_t0_data")
            wt_predicted_shading = pd.read_csv('data/raw/Shading' + index + '.csv',
                                               header=None)  # rows: m of stream, columns: timesteps in min
            wt_predicted_shading = wt_predicted_shading.iloc[measurement_points["Distance (m)"]]
            wt_predicted_shading = wt_predicted_shading.iloc[:, ::15].transpose()
            wt_predicted_shading.columns = ["wt_shading_predicted_point_" + str(i) for i in
                                            measurement_points["Distance (m)"]]
            # fix index columns
            x0_data.index = wt_observed.index
            wt_predicted.index = wt_observed.index
            discharge_805.index = wt_observed.index
            wt_predicted_shading.index = wt_observed.index
            # concat data
            data_sub = pd.concat([met_data, precip.iloc[:, 1], discharge_805,
                                  wt_observed, wt_predicted, wt_predicted_shading, x0_data], axis=1)
            append_data.append(data_sub)

        # Concatenate full data set
        data = pd.concat(append_data)
        data_time_index = pd.DataFrame({'year': data.Year.tolist(),
                                        'month': data.Month.tolist(),
                                        'hour': data.Hour.tolist(),
                                        'minute': data.Minute.tolist(),
                                        'day': data.Day.tolist()})
        data.index = pd.to_datetime(data_time_index)
        data = data.sort_index()
        # Define training/validation
        validation_timesteps = 4 * 24 * days_for_validation
        cal_ts = len(data.index) - validation_timesteps
        if random_validation:
            cal_val = ["calibration" for i in range(cal_ts)] + ["validation" for i in range(validation_timesteps)]
            shuffled_index = np.random.RandomState(seed=seed).permutation(len(cal_val)).tolist()
            cal_val = [cal_val[i] for i in shuffled_index]
        else:
            #cal_val = ["calibration" for x in range(cal_ts)] + ["validation" for x in range(validation_timesteps)]
            cal_val = ["validation" for x in range(validation_timesteps)] + ["calibration" for x in range(cal_ts)]

        data['calibration_validation'] = pd.Series(cal_val, index=data.index)
        # Compute residual columns
        for point in measurement_points["Distance (m)"]:
            data['residuals_point_' + str(point)] = data['wt_observed_point_' + str(point)] - data[
                'wt_predicted_point_' + str(point)]
            data['shading_residuals_point_' + str(point)] = data['wt_observed_point_' + str(point)] - data[
                'wt_shading_predicted_point_' + str(point)]
        # Save as csv
        print('Finished preprocessing. Final data set is stored in "data/preprocessed/data.csv"')
        data['sin_hour'] = np.sin(2 * np.pi * data.Hour / 24)
        data['cos_hour'] = np.cos(2 * np.pi * data.Hour / 24)

        # create lagged features
        lag_variables = ['Shortwave Radiation (W/m2)',
                         'Air Temperature (deg C)',
                         'Relative Humidity (%)',
                         'Wind Speed (m/s)',
                         'Longwave radiation',
                         'Precip (mm)',
                         'Discharge (m3/s)',
                         'Tree Temp']
        data = create_lags(data, lag_variables, lag)
        data.to_csv("data/processed/data.csv", index_label=False)
        # Data for ML models
        model_variables = ['sin_hour', 'cos_hour',
                           'Shortwave Radiation (W/m2)',
                           'Air Temperature (deg C)',
                           'Relative Humidity (%)',
                           'Wind Speed (m/s)',
                           'Longwave radiation',
                           'Precip (mm)',
                           'Discharge (m3/s)',
                           'Tree Temp']

        # training data
        training_data = data[data["calibration_validation"] != "validation"]
        x_train = training_data[model_variables]
        y_train = training_data['residuals_point_640']
        # Validation data
        validation_data = data[data["calibration_validation"] == "validation"]
        x_val = validation_data[model_variables]
        y_val = validation_data['residuals_point_640']
        # full dataset x, y
        x = data[model_variables]
        y = data['residuals_point_640']
        # Save as csv
        x_train.to_csv("data/processed/x_train.csv", index_label=False)
        y_train.to_csv("data/processed/y_train.csv", index_label=False)
        x_val.to_csv("data/processed/x_val.csv", index_label=False)
        y_val.to_csv("data/processed/y_val.csv", index_label=False)
        x.to_csv("data/processed/x.csv", index_label=False)
        y.to_csv("data/processed/y.csv", index_label=False)
    if not random_validation:
        print("Time periods")
        training_data = data[data["calibration_validation"] != "validation"]
        validation_data = data[data["calibration_validation"] == "validation"]
        print(f"Training: {training_data.index[0]} - {training_data.index[-1]}")
        print(f"Validation:  {validation_data.index[0]} - {validation_data.index[-1]}")
    return data, x_train, y_train, x_val, y_val, x, y


def residual_plots(data):
    """
    Descriptive plots for model residuals
    Plots are saved under results/

    Parameters
    ----------
    data: data frame
    """
    # Residual Histograms
    plt.subplots(2, 5, figsize=(30, 15))
    # sns.set(rc={'figure.figsize': (45, 1)})
    plot_data = data[['residuals_point_0', 'residuals_point_45',
                      'residuals_point_196', 'residuals_point_223', 'residuals_point_324',
                      'residuals_point_356', 'residuals_point_494', 'residuals_point_640',
                      'residuals_point_771', 'residuals_point_805']]
    for i, col in enumerate(plot_data.columns):
        plt.subplot(2, 5, i + 1)
        sns.kdeplot(plot_data[col], shade=True, legend=False)
        plt.ylabel('model residuals')
        plt.xlim(-2, 2)
        plt.title(str(plot_data.columns[i]))
        # plt.subplots_adjust(right=1.5, top=0.9)
    plt.savefig('results/figures/01_residual_hist.png')
    plt.clf()

    # Residual Correlations
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

    # Autocorrelation of point 640 residuals
    sns.set(rc={'figure.figsize': (30, 20)})
    plot = pd.plotting.autocorrelation_plot(data['residuals_point_640'])
    plot.figure.savefig('results/figures/03_residual_autocorr_point640.png')

    # Model residuals vs shading residuals in August
    plot_data_all_aug = data.loc[data["Month"] == 8]
    plot_data_aug = plot_data_all_aug[
        ['residuals_point_223', 'shading_residuals_point_223']]
    sns.set(rc={'figure.figsize': (24, 7)})
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.5)
    plot = plot_data_aug.plot(linewidth=2, legend=False)
    plot.legend(title='', loc='upper right', labels=['Model residuals', 'Shading residuals'])
    plot.figure.savefig('results/figures/04_prediction_vs_shading_residuals_point640.png')


def eda_plots(data, model_variables, point):
    """
    Explorative data analysis (EDA) plots of given data frame and variables
    Plots are saved under results/

    Parameters
    ----------
    data: data frame
    model_variables: list of characters including model variables
    point: measurement point observations which should be used
    """

    sns.set(rc={'figure.figsize': (24, 7)})
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.5)
    plot_data_all_aug = data.loc[data["Month"] == 8]
    plot_data_aug = plot_data_all_aug[
        ['wt_observed_point_' + point, 'wt_predicted_point_' + point, 'wt_shading_predicted_point_' + point]]
    plot = plot_data_aug.plot(linewidth=2, legend=False)
    plot.legend(title='', loc='upper right', labels=['Observation', 'Model prediction', 'Shading'])
    rmse = np.sqrt(
        mean_squared_error(plot_data_aug['wt_predicted_point_' + point], plot_data_aug['wt_observed_point_' + point]))
    shading_rmse = np.sqrt(
        mean_squared_error(plot_data_aug['wt_shading_predicted_point_' + point],
                           plot_data_aug['wt_observed_point_' + point]))
    plot.set_title(
        f'Predicted and observed stream water temperature, RMSE = {rmse:.4} °C, RMSE shading = {shading_rmse:.4} °C')
    plot.figure.savefig('results/figures/05_wt_timeseries_August_with_shading.png')
    plt.close()

    sns.set(rc={'figure.figsize': (24, 7)})
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.5)
    plot_data_all_aug = data.loc[data["Month"] == 8]
    plot_data_all_aug.index = pd.to_datetime(plot_data_all_aug.index)
    plot_data_aug = plot_data_all_aug[
        ['wt_observed_point_' + point, 'wt_predicted_point_' + point]]
    plot = plot_data_aug.plot(linewidth=2, legend=False)
    plot.legend(title='', loc='upper right', labels=['Observation', 'Model prediction'])
    plot.set(ylabel='Stream water temperature (°C)')

    rmse = np.sqrt(
        mean_squared_error(plot_data_aug['wt_predicted_point_' + point], plot_data_aug['wt_observed_point_' + point]))
    plot.set_title(
        f'Predicted and observed stream water temperature, RMSE = {rmse:.4} °C')
    plot.figure.savefig('results/figures/05_wt_timeseries_August.png')

    # Feature correlation
    sns.set(rc={'figure.figsize': (23, 21)})
    corr = data[model_variables].corr().round(2)
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    # Set up the matplotlib figure
    sns.set_context("notebook", font_scale=1)
    #plt.subplots(figsize=(24, 24))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.set(font_scale=1.4)
    g = sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
                square=True, linewidths=.9, cbar_kws={"shrink": .5},
                annot=True)
    #g.set_xticklabels(g.get_xmajorticklabels(), fontsize=20)
    #g.set_yticklabels(g.get_ymajorticklabels(), fontsize=20)
    plt.savefig('results/figures/06_feature_correlation.png')
    plt.close('all')