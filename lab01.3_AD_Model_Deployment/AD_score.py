import pandas as pd
from pyculiarity import detect_ts
import os
# from pytictoc import TicToc
# from pyculiarity.date_utils import date_format
# import matplotlib.pyplot as plt
# import seaborn
# import datetime
import numpy as np
import json
import glob

def create_data_dict(data, sensors):
    """

    :param data:
    :return:
    """
    data_dict = {}
    for column in data.columns:
        data_dict[column] = [data[column].values[0]]
        if column in sensors:
            data_dict[column + '_avg'] = [0.0]
            data_dict[column + '_an'] = [False]

    return data_dict


def init_df(data):
    """
    Init DataFrame from one row of data
    :param data:
    :return:
    """

    # data_dict = create_data_dict(data)

    df = pd.DataFrame() #data=data_dict, index=data_dict['timestamp'])

    return df


def append_data(df, data, sensors):
    """
    We either add the data and the results (res_dict) of the anomaly detection to the existing data frame,
    or create a new one if the data frame is empty
    """
    data_dict = create_data_dict(data, sensors)

    df = df.append(pd.DataFrame(data=data_dict, index=data_dict['timestamp']))

    return df


def init():
    global storage_location

    storage_location = "/tmp/output"

    if not os.path.exists(storage_location):
        os.makedirs(storage_location)

    # next, we delete previous output files
    files = glob.glob(os.path.join(storage_location,'*'))
    
    for f in files:
        os.remove(f)
        
        
def generate_stream(telemetry, n=None):
    """
        n is the number of sensor readings we are simulating
        """

    if not n:
        n = telemetry.shape[0]

    machine_ids = [1] # telemetry['machineID'].unique()
    timestamps = telemetry['timestamp'].unique()

    # sort test_data by timestamp
    # on every iteration, shuffle machine IDs
    # then loop over machine IDs

    #t = TicToc()
    for timestamp in timestamps:
        #t.tic()
        np.random.shuffle(machine_ids)
        for machine_id in machine_ids:
            data = telemetry.loc[(telemetry['timestamp'] == timestamp) & (telemetry['machineID'] == machine_id), :]
            run(data)
        #t.toc("Processing all machines took")


def load_df(data):
    machineID = data['machineID'].values[0]

    filename = os.path.join(storage_location, "data_w_anoms_ID_%03d.csv" % machineID)
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%d %H:%M:%S")
    else:
        df = init_df(data)

    return df


def save_df(df):
    """

    :param df:
    :return:
    """
    machine_id = df.ix[0, 'machineID']

    filename = os.path.join(storage_location, "data_w_anoms_ID_%03d.csv" % machine_id)

    df.to_csv(filename, index=False)


def running_avgs(df, sensors, window_size=24):
    """
    Calculates rolling average according to Welford's online algorithm.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online

    This adds a column next to the column of interest, with the suffix '_<n>' on the column name

    :param df: a dataframe with time series in columns
    :param column: name of the column of interest
    :param n: number of measurements to consider
    :return: None
    """

    curr_n = df.shape[0]
    row_index = curr_n - 1
    window_size = min(window_size, curr_n)

    for sensor in sensors:
        val_col_index = df.columns.get_loc(sensor)
        avg_col_index = df.columns.get_loc(sensor + "_avg")

        curr_value = df.ix[row_index, val_col_index]

        if curr_n > 1:
            prv_avg = df.ix[(row_index -1), avg_col_index]
            df.ix[row_index, avg_col_index] = prv_avg + (curr_value - prv_avg) / window_size
        else:
            df.ix[row_index, avg_col_index] = curr_value


def run(data, window=14 * 24):
    """

    :param data:
    :param window:
    :return:
    """

    # set some parameters for the AD algorithm
    alpha = 0.1
    max_anoms = 0.05
    only_last = None  # alternative, we can set this to 'hr' or 'day'

    data = pd.read_json(json.loads(data)['data'])
    # return json.dumps(data.columns)

    sensors = ['volt','pressure','vibration', 'rotate']  # list(data.columns[2:])

    # load dataframe
    df = load_df(data)

    # add current sensor readings to data frame, also adds fields for anomaly detection results
    df = append_data(df, data, sensors)

    # calculate running averages
    running_avgs(df, sensors)

    # note timestamp  so that we can update the correct row of the dataframe later
    timestamp = data['timestamp'].values[0]

    # we get a copy of the current (also last) row of the dataframe
    current_row = df.loc[df['timestamp'] == timestamp, :]

    
    # determine how many sensor readings we already have
    rows = df.shape[0]

    # if the data frame doesn't have enough rows for our sliding window size, we just return (setting that we have no
    #  anomalies)
    if rows < window:
        save_df(df)
        json_data = current_row.to_json()
        
        return json.dumps(json_data)

    
    # determine the first row of the data frame that falls into the sliding window
    start_row = rows - window

    # a flag to indicate whether we detected an anomaly in any of the sensors after this reading
    detected_an_anomaly = False

    # we loop over the sensor columns
    for column in sensors:
        df_s = df.ix[start_row:rows, ('timestamp', column + "_avg")]

        # pyculiarity expects two columns with particular names
        df_s.columns = ['timestamp', 'value']

        # we reset the timestamps, so that the current measurement is the last within the sliding time window
        # df_s = reset_time(df_s)

        # calculate the median value within each time sliding window
        # values = df_s.groupby(df_s.index.date)['value'].median()

        # create dataframe with median values etc.
        # df_agg = pd.DataFrame(data={'timestamp': pd.to_datetime(values.index), 'value': values})

        # find anomalies
        results = detect_ts(df_s, max_anoms=max_anoms,
                            alpha=alpha,
                            direction='both',
                            e_value=False,
                            only_last=only_last)

        # create a data frame where we mark for each day whether it was an anomaly
        df_s = df_s.merge(results['anoms'], on='timestamp', how='left')

        # mark the current sensor reading as anomaly Specifically, if we get an anomaly in the the sliding window
        # leading up (including) the current sensor reading, we mark the current sensor reading as anomaly note,
        # alternatively one could mark all the sensor readings that fall within the sliding window as anomalies.
        # However, we prefer our approach, because without the current sensor reading the other sensor readings in
        # this sliding window may not have been an anomaly
        # current_row[column + '_an'] = not np.isnan(df_agg.tail(1)['anoms'].iloc[0])
        if not np.isnan(df_s.tail(1)['anoms'].iloc[0]):
            current_row.ix[0,column + '_an'] = True
            detected_an_anomaly = True

    # It's only necessary to update the current row in the data frame, if we detected an anomaly
    if detected_an_anomaly:
        df.loc[df['timestamp'] == timestamp, :] = current_row
    save_df(df)
    
    json_data = current_row.to_json()
    
    return json.dumps(json_data)


def main():
    print("Reading data ... ", end="")
    telemetry = pd.read_csv(os.path.join('data', 'telemetry.csv'))  # , parse_dates=['datetime'])
    print("Done.")

    print("Parsing datetime...", end="")
    telemetry['datetime'] = pd.to_datetime(telemetry['datetime'], format="%m/%d/%Y %I:%M:%S %p")
    telemetry.columns = ['timestamp', 'machineID', 'volt', 'rotate', 'pressure', 'vibration']
    print("Done.")

    print("Processing ... ", end="")
    init()
    generate_stream(telemetry)
    print("Done.")


if __name__ == "__main__":
    main()

