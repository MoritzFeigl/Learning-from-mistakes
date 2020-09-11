# Imports
import pandas as pd
import numpy as np
from os.path import isfile
def load_preprocess_data(days_for_validation, random_validation=False, seed=None):
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
    if isfile('data/processed/data.csv'):
        print('Load previously computed data set from "data/preprocessed/data.csv"')
        data = pd.read_csv('data/processed/data.csv')
        x_train = pd.to_csv("data/processed/x_train.csv")
        y_train = pd.to_csv("data/processed/y_train.csv")
        x_val = pd.to_csv("data/processed/x_val.csv")
        y_val = pd.to_csv("data/processed/x_val.csv")
    else:

        append_data = []
        for index in ['C', 'V', 'V2', 'V3']:
            # Meteorological Data
            met_data = pd.read_excel('data/raw/Input' + index + '.xlsx', sheet_name = "met_data")
            precip = pd.read_excel('data/raw/Input' + index + '.xlsx', sheet_name = "precip")
            dis_data = pd.read_excel('data/raw/Input' + index + '.xlsx', sheet_name = "dis_data", skiprows=1, header=None)
            discharge_805 = pd.DataFrame({'Discharge (m3/s)': dis_data.iloc[4,1:].transpose()})
            # observed wt
            wt_observed = pd.read_excel('data/raw/Input' + index + '.xlsx', sheet_name = "temp", header = None).transpose()
            measurement_points = pd.read_excel('data/raw/Input' + index + '.xlsx', sheet_name = "temp_t0_data")
            wt_observed.columns = ["wt_observed_point_" + str(i) for i in measurement_points["Distance (m)"]]
            # observed wt at boundary
            x0_data = pd.read_excel('data/raw/Input' + index + '.xlsx', sheet_name = "temp_x0_data")
            x0_data = x0_data.drop(labels='Time (min)', axis=1)
            x0_data.columns = ['x0 Temperature (deg C)']
            # predicted wt
            wt_predicted = pd.read_csv('data/raw/Output' + index + '.csv', header = None) # rows: m of stream, columns: timesteps in min
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
            wt_predicted_shading.columns = ["wt_shading_predicted_point_" + str(i) for i in measurement_points["Distance (m)"]]
            # fix index columns
            x0_data.index = wt_observed.index
            wt_predicted.index = wt_observed.index
            discharge_805.index = wt_observed.index
            wt_predicted_shading.index = wt_observed.index
            # concat data
            data_sub = pd.concat([met_data, precip.iloc[:, 1], discharge_805,
                                  wt_observed, wt_predicted, wt_predicted_shading, x0_data], axis = 1)
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
        validation_timesteps = 4*24*days_for_validation
        cal_ts = len(data.index) -  validation_timesteps
        if random_validation:
            cal_val = ["calibration" for x in range(cal_ts)] + ["validation" for x in range(validation_timesteps)]
            shuffled_index = np.random.RandomState(seed=seed).permutation(len(cal_val)).tolist()
            cal_val = [cal_val[i] for i in shuffled_index]
        else:
            cal_val = ["calibration" for x in range(cal_ts)] + ["validation" for x in range(validation_timesteps)]
        data['calibration_validation'] = pd.Series(cal_val, index=data.index)
        # Compute residual columns
        for point in measurement_points["Distance (m)"]:
            data['residuals_point_' + str(point)] = data['wt_observed_point_' + str(point)] - data[
                'wt_predicted_point_' + str(point)]
            data['shading_residuals_point_' + str(point)] = data['wt_observed_point_' + str(point)] - data[
                'wt_shading_predicted_point_' + str(point)]
        # Save as csv
        data.to_csv("data/processed/data.csv")
        print('Finished preprocessing. Final data set is stored in "data/preprocessed/data.csv"')
        data['sin_hour'] = np.sin(2 * np.pi * data.Hour / 24)
        data['cos_hour'] = np.cos(2 * np.pi * data.Hour / 24)
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
        x_train.to_csv("data/processed/x_train.csv")
        y_train.to_csv("data/processed/y_train.csv")
        x_val.to_csv("data/processed/x_val.csv")
        y_val.to_csv("data/processed/x_val.csv")
        x.to_csv("data/processed/x.csv")
        y.to_csv("data/processed/y.csv")
    if not random_validation:
        print("Time periods")
        print(f"Training: {training_data.index[0]} - {training_data.index[-1]}")
        print(f"Validation:  {validation_data.index[0]} - {validation_data.index[-1]}")
    return data, x_train, y_train, x_val, y_val, x, y