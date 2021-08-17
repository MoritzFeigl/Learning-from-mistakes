# Imports
import os
from os.path import isfile
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import re
from typing import Tuple


def load_preprocess_data(days_for_validation: int,
                         lag_variables: list,
                         random_validation: bool = False,
                         seed: int = None,
                         lag: int = 8,
                         reload: bool = True,
                         save_csv: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                         pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loading and data preprocessing for the Stream water temperature case study

    Parameters
    ----------
    days_for_validation : int
        Number of days used for validation
    lag_variables : list[str]
        List with variable names that should be lagged
    random_validation :
    seed : int
        Random seed. Only relevant if random_validation=True
    lag : int
        number of lagged time steps that are computed for all lag_variables.
    reload : bool
        Should a previously computed processed data set be loaded? True/False
    save_csv : bool
        Should the preprocessed data be saved as a csv? Necessary if reload=True will be used.

    Returns
    -------
    Tuple of pd.DataFrames:
        data : Full preprocessed data set
        x_train : Training features
        y_train : Training labels
        x_test : Test features
        y_test :
        x : All features
        y : All labels
    """
    if isfile('data/processed/data.csv') and reload:
        print('Load previously computed data set from "data/preprocessed/data.csv"')
        data = pd.read_csv('data/processed/data.csv')
        x_train = pd.read_csv("data/processed/x_train.csv")
        y_train = pd.read_csv("data/processed/y_train.csv")
        x_test = pd.read_csv("data/processed/x_test.csv")
        y_test = pd.read_csv("data/processed/y_test.csv")
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

            # fix index columns
            x0_data.index = wt_observed.index
            wt_predicted.index = wt_observed.index
            discharge_805.index = wt_observed.index
            # concat data
            data_sub = pd.concat([met_data, precip.iloc[:, 1], discharge_805,
                                  wt_observed, wt_predicted, x0_data], axis=1)
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
        # Define training/validation column
        validation_timesteps = 4 * 24 * days_for_validation
        cal_ts = len(data.index) - validation_timesteps
        if random_validation:
            cal_val = ["calibration" for i in range(cal_ts)] + ["validation" for i in range(validation_timesteps)]
            shuffled_index = np.random.RandomState(seed=seed).permutation(len(cal_val)).tolist()
            cal_val = [cal_val[i] for i in shuffled_index]
        else:
            # cal_val = ["calibration" for x in range(cal_ts)] + ["validation" for x in range(validation_timesteps)]
            cal_val = ["validation" for x in range(validation_timesteps)] + ["calibration" for x in range(cal_ts)]
        data['calibration_validation'] = pd.Series(cal_val, index=data.index)

        # Compute residual columns
        for point in measurement_points["Distance (m)"]:
            data['residuals_point_' + str(point)] = data['wt_predicted_point_' + str(point)] - \
                                                    data['wt_observed_point_' + str(point)]

        # Save as csv
        data['sin_hour'] = np.sin(2 * np.pi * data.Hour / 24)
        data['cos_hour'] = np.cos(2 * np.pi * data.Hour / 24)
        # remove dupolicated rows if any exist
        data = data[~data.index.duplicated(keep='first')]
        # create lagged features
        data = create_lags(data, lag_variables, lag)

        # Data for ML models
        lagged_variable_names = [[x + "_lag" + str(y + 1) for y in range(lag)] for x in lag_variables]
        model_variables = ['sin_hour', 'cos_hour'] + lag_variables + sum(lagged_variable_names, [])

        # training data
        training_data = data[data["calibration_validation"] != "validation"]
        x_train = training_data[model_variables]
        y_train = training_data['residuals_point_640']
        # Validation data
        validation_data = data[data["calibration_validation"] == "validation"]
        x_test = validation_data[model_variables]
        y_test = validation_data['residuals_point_640']
        # full dataset x, y
        x = data[model_variables]
        y = data['residuals_point_640']
        # Save as csv
        if save_csv:
            data.to_csv("data/processed/data.csv", index_label=False)
            x_train.to_csv("data/processed/x_train.csv", index_label=False)
            y_train.to_csv("data/processed/y_train.csv", index_label=False)
            x_test.to_csv("data/processed/x_test.csv", index_label=False)
            y_test.to_csv("data/processed/y_test.csv", index_label=False)
            x.to_csv("data/processed/x.csv", index_label=False)
            y.to_csv("data/processed/y.csv", index_label=False)
            print('Finished preprocessing. Final data sets are stored in "data/preprocessed/"')
    if not random_validation:
        print("Time periods")
        training_data = data[data["calibration_validation"] != "validation"]
        validation_data = data[data["calibration_validation"] == "validation"]
        print(f"Training: {training_data.index[0]} - {training_data.index[-1]}")
        print(f"Validation:  {validation_data.index[0]} - {validation_data.index[-1]}")
    return data, x_train, y_train, x_test, y_test, x, y


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
    return x_with_lags


def residual_plots(data: pd.DataFrame):
    """ Plot HFLUX residuals of stream temperature measurement points

    Parameters
    ----------
    data : pd.DataFrame
        Data produced by load_preprocess_data() including residuals_point_XX columns.
    """
    os.makedirs("results/figures/04_prediction_residuals", exist_ok=True)
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

    sns.clustermap(corr, method="complete", cmap='RdBu', annot=True,
                   annot_kws={"size": 15}, vmin=-1, vmax=1, figsize=(15, 12))
    plt.savefig('results/figures/02_residual_corr.png')
    plt.close()
    print("Correlation plot of stream temperature measurement points is saved "
          "under results/figures/02_residual_corr.png")

    # Autocorrelation of point 640 residuals
    plot = pd.plotting.autocorrelation_plot(data['residuals_point_640'])
    plot.figure.savefig('results/figures/03_residual_autocorr_point640.png')

    # New shading & canopy temperature residuals
    append_data = []
    for index in ['C', 'V', 'V3']:
        met_data = pd.read_excel('data/raw/Input' + index + '.xlsx', sheet_name="met_data")
        measurement_points = pd.read_excel('data/raw/Input' + index + '.xlsx', sheet_name="temp_t0_data")
        wt_predicted_sc = pd.read_csv(
            'data/raw/canopy_temp_and_shading_1200-1430-1700_no_calibration/Output' + index + '.csv',
            header=None)  # rows: m of stream, columns: timesteps in min
        wt_predicted_sc = wt_predicted_sc.iloc[measurement_points["Distance (m)"]]
        wt_predicted_sc = wt_predicted_sc.iloc[:, ::15].transpose()
        wt_predicted_sc.index = met_data.index
        wt_predicted_sc.columns = ["wt_sc_predicted_point_" + str(i) for i in
                                   measurement_points["Distance (m)"]]
        append_data.append(pd.concat([met_data, wt_predicted_sc], axis=1))
    shading_canopy_data = pd.concat(append_data)
    sc_data_time_index = pd.DataFrame({'year': shading_canopy_data.Year.tolist(),
                                       'month': shading_canopy_data.Month.tolist(),
                                       'hour': shading_canopy_data.Hour.tolist(),
                                       'minute': shading_canopy_data.Minute.tolist(),
                                       'day': shading_canopy_data.Day.tolist()})
    shading_canopy_data.index = pd.to_datetime(sc_data_time_index)
    shading_canopy_data = shading_canopy_data.sort_index()
    # remove dupolicated rows if any exist
    shading_canopy_data = shading_canopy_data[~shading_canopy_data.index.duplicated(keep='first')]
    remove_rows = shading_canopy_data.shape[0] - data.shape[0]
    shading_canopy_data = shading_canopy_data.iloc[remove_rows:, :]  # remove lag times
    shading_canopy_data.index = data.index
    plot_data_sc = pd.concat([data[['wt_observed_point_640', 'wt_predicted_point_640']],
                              shading_canopy_data['wt_sc_predicted_point_640']], axis=1)
    plot_data_sc["residuals_point_640"] = plot_data_sc["wt_predicted_point_640"] - plot_data_sc["wt_observed_point_640"]
    plot_data_sc["sc_residuals"] = plot_data_sc["wt_sc_predicted_point_640"] - plot_data_sc["wt_observed_point_640"]
    plot_data_sc = plot_data_sc.drop(
        columns=["wt_observed_point_640", "wt_sc_predicted_point_640", "wt_predicted_point_640"])
    plot = plot_data_sc.plot(linewidth=2, legend=False)
    plt.legend(title='', loc='upper right', fontsize=18,
               labels=['HFLUX Residuals', 'New Shading & Canopy\nTemperature Residuals'])
    plot.tick_params(axis='x', labelrotation=45)
    plot.set_ylabel('Stream Temperature Residuals (°C)', fontsize=24)
    plot.tick_params(axis='x', labelsize=18)
    plot.tick_params(axis='y', labelsize=18)
    plot.set_xlabel('', fontsize=24)
    plot.figure.savefig('results/figures/04_prediction_residuals/04_residuals_HFLUX_vs_ShadingCanopyTemp.png',
                        bbox_inches='tight', dpi=600)
    plt.close()
    print("Model residuals of model using new canopy temperature and new shading is saved "
          "under results/figures/04_prediction_residuals/04_residuals_HFLUX_vs_ShadingCanopyTemp.png")


def data_plots(data: pd.DataFrame,
               model_variables: list,
               point: str):
    """Plots of model variables and prediction results

    Parameters
    ----------
    data: data frame
    model_variables: list of characters including model variables
    point: measurement point observations which should be used
    Parameters
    ----------
    data : pd.DataFrame
        Data produced by load_preprocess_data().
    model_variables : list[str]
        List of variables used as inputs(i.e. features) for the error ml-model
    point : str
        Number of the chosen stream temperature measurement point.
    """

    os.makedirs("results/figures/05_prediction_plots", exist_ok=True)

    # 1. Plot: Time series of observed and predicted water temperature
    sns.set(rc={'figure.figsize': (24, 7)})
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.5)
    plot_data_all = data.copy()
    plot_data_all.index = pd.to_datetime(plot_data_all.index)
    plot_data_aug = plot_data_all[
        ['wt_observed_point_' + point, 'wt_predicted_point_' + point]]
    plot = plot_data_aug.plot(linewidth=2, legend=False)
    plot.legend(title='', loc='upper right', labels=['Observation', 'HFLUX prediction'])
    plot.set(ylabel='Stream water temperature (°C)')

    rmse = np.sqrt(
        mean_squared_error(plot_data_aug['wt_predicted_point_' + point], plot_data_aug['wt_observed_point_' + point]))
    plot.set_title(
        f'Predicted and observed stream water temperature, RMSE = {rmse:.3} °C')
    plot.figure.savefig('results/figures/05_prediction_plots/05_observed_vs_HFLUX.png',
                        dpi=600, bbox_inches='tight')
    print("Plot of Predicted and observed stream water temperature is saved "
          "under results/figures/05_prediction_plots/05_observed_vs_HFLUX.png")

    # 2. Plot:plot 1 + shading & new canopy temperature results
    # get predictions
    append_data = []
    for index in ['C', 'V', 'V3']:
        met_data = pd.read_excel('data/raw/Input' + index + '.xlsx', sheet_name="met_data")
        measurement_points = pd.read_excel('data/raw/Input' + index + '.xlsx', sheet_name="temp_t0_data")
        wt_predicted_sc = pd.read_csv(
            'data/raw/canopy_temp_and_shading_1200-1430-1700_no_calibration/Output' + index + '.csv',
            header=None)  # rows: m of stream, columns: timesteps in min
        wt_predicted_sc = wt_predicted_sc.iloc[measurement_points["Distance (m)"]]
        wt_predicted_sc = wt_predicted_sc.iloc[:, ::15].transpose()
        wt_predicted_sc.index = met_data.index
        wt_predicted_sc.columns = ["wt_sc_predicted_point_" + str(i) for i in
                                   measurement_points["Distance (m)"]]
        append_data.append(pd.concat([met_data, wt_predicted_sc], axis=1))
    shading_canopy_data = pd.concat(append_data)
    sc_data_time_index = pd.DataFrame({'year': shading_canopy_data.Year.tolist(),
                                       'month': shading_canopy_data.Month.tolist(),
                                       'hour': shading_canopy_data.Hour.tolist(),
                                       'minute': shading_canopy_data.Minute.tolist(),
                                       'day': shading_canopy_data.Day.tolist()})
    shading_canopy_data.index = pd.to_datetime(sc_data_time_index)
    shading_canopy_data = shading_canopy_data.sort_index()

    # remove dupolicated rows if any exist
    shading_canopy_data = shading_canopy_data[~shading_canopy_data.index.duplicated(keep='first')]
    remove_rows = shading_canopy_data.shape[0] - data.shape[0]
    shading_canopy_data = shading_canopy_data.iloc[remove_rows:, :]  # remove lag times
    shading_canopy_data.index = data.index
    plot_data_sc = pd.concat([data[['wt_observed_point_640', 'wt_predicted_point_' + point]],
                              shading_canopy_data['wt_sc_predicted_point_' + point]], axis=1)
    plot_data_sc.index = pd.to_datetime(plot_data_sc.index).strftime('%Y-%m-%d')
    sns.set(rc={'figure.figsize': (24, 7)})
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.5)
    plot = plot_data_sc.plot(linewidth=2, legend=False)
    plot.set_ylabel('Stream Temperature (°C)', fontsize=18)
    plot.tick_params(axis='x', labelsize=18)
    plot.tick_params(axis='y', labelsize=18)
    plot.legend(title='', loc='upper right',
                labels=['Observation', 'HFLUX Prediction', 'New Shading & Canopy\nTemperature Prediction'])
    plt.xticks(rotation=45)
    rmse = np.sqrt(mean_squared_error(plot_data_sc['wt_predicted_point_' + point],
                                      plot_data_sc['wt_observed_point_' + point]))
    sc_rmse = np.sqrt(mean_squared_error(plot_data_sc['wt_sc_predicted_point_' + point],
                                         plot_data_sc['wt_observed_point_' + point]))
    plot.set_title(
        f'Predicted and observed stream water temperature, RMSE = {rmse:.3} °C, RMSE Shading & Canopy Temperature = {sc_rmse:.3} °C')
    plot.figure.savefig('results/figures/05_prediction_plots/05_HFLUX_vs_ShadingCanopyTemp.png',
                        dpi=600, bbox_inches='tight')
    plt.close()
    print("Plot of Predicted, observed and new shading/canopy temperature stream water temperature is saved "
          "under results/figures/05_prediction_plots/05_observed_vs_HFLUX.png")

    # compute RMSE for nights
    data_sc_hours = pd.concat([data[['Hour', 'wt_observed_point_' + point, 'wt_predicted_point_' + point]],
                               shading_canopy_data['wt_sc_predicted_point_' + point]], axis=1)
    data_sc_night = data_sc_hours[data_sc_hours.Hour.isin([22, 23, 0, 1, 2, 3, 4, 5, 6])]
    rmse_nigth = np.sqrt(mean_squared_error(data_sc_night['wt_predicted_point_' + point],
                                            data_sc_night['wt_observed_point_' + point]))
    sc_rmse_nigth = np.sqrt(mean_squared_error(data_sc_night['wt_sc_predicted_point_' + point],
                                               data_sc_night['wt_observed_point_' + point]))
    print(f"Night time (22-6) RMSE changed from {rmse_nigth:.3} to {sc_rmse_nigth:.3} "
          f"using new shading and measured canopy temperature")
    # compute RMSE for times of changed shading
    data_sc_shade = data_sc_hours[data_sc_hours.Hour.isin([12, 13, 14, 15, 16, 17])]
    rmse_shade = np.sqrt(mean_squared_error(data_sc_shade['wt_predicted_point_' + point],
                                            data_sc_shade['wt_observed_point_' + point]))
    sc_rmse_shade = np.sqrt(mean_squared_error(data_sc_shade['wt_sc_predicted_point_' + point],
                                               data_sc_shade['wt_observed_point_' + point]))
    print(
        f"Afternoon (12-17) RMSE changed from {rmse_shade:.3} to {sc_rmse_shade:.3} "
        f"using new shading and measured canopy temperature")

    # 5. Plot: Feature correlation
    sns.set(rc={'figure.figsize': (14, 12)})
    relevant_vars = data[model_variables[2:]]
    relevant_vars.columns = list(map(lambda v: re.split(' \(', v)[0], model_variables[2:]))
    corr = relevant_vars.corr().round(2)
    sns.clustermap(corr, method="complete", cmap='RdBu', annot=True,
                   annot_kws={"size": 24}, vmin=-1, vmax=1, figsize=(15, 12))
    plt.savefig('results/figures/06_feature_correlation.png', dpi=600)
    plt.close('all')
    print("Feature correlation plot is saved "
          "under results/figures/06_feature_correlation.png")


def create_pca_features(x: pd.DataFrame,
                        x_train: pd.DataFrame,
                        x_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Compute PCA of given full/training/test data
    PCA is fitted on the full dataset x and then applied to x, x_train, x_val.

    Parameters
    ----------
    x : pd.DataFrame
        All features
    x_train : pd.DataFrame
        Training features
    x_test : pd.DataFrame
        Test features

    Returns
    -------
     Tuple of pd.DataFrames:
        x_pca: Principal components of x
        x_pca_train: Principal components of x_train
        x_pca_val: Principal components of x_train
        loadings: Loadings of Principal components

    """
    scaler = StandardScaler()
    scaler.fit(x)
    x_scaled, x_train_scaled, x_test_scaled = scaler.transform(x), scaler.transform(x_train), scaler.transform(x_test)
    # compute PCA
    pca = PCA()
    pca.fit(x_scaled)
    pc_names = ["PC" + str(x + 1) for x in range(pca.components_.shape[0])]
    loadings = pd.DataFrame(pca.components_.T, columns=pc_names, index=x.columns)
    # define new x matrices
    x_pca = pd.DataFrame(pca.transform(x_scaled))
    x_pca_train = pd.DataFrame(pca.transform(x_train_scaled))
    x_pca_val = pd.DataFrame(pca.transform(x_test_scaled))
    x_pca.index, x_pca_train.index, x_pca_val.index = x.index, x_train.index, x_test.index
    x_pca.columns = x_pca_train.columns = x_pca_val.columns = pc_names
    return x_pca, x_pca_train, x_pca_val, loadings
