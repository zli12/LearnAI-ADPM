
def sample_run(df, window_size = 500, com = 12):
    """
    This functions expects a dataframe df as mandatory argument.  
    The first column of the df should contain timestamps, the second machine IDs
    
    Keyword arguments:
    n_machines_test: the number of machines to include in the sample
    ts_per_machine: the number of timestamps to test for each machine
    window_size: the size of the window of data points that are used for anomaly detection
    
    """

    n_epochs = 10
    p_anoms = .5
    
    # create arrays that will hold the results of batch AD (y_true) and online AD (y_pred)
    y_true = []
    y_pred = []
    run_times = []
    
    # check which unique machines, sensors, and timestamps we have in the dataset
    machineIDs = df['machineID'].unique()
    sensors = df.columns[2:]
    timestamps = df['datetime'].unique()[window_size:]
    
    # sample n_machines_test random machines and sensors 
    random_machines = np.random.choice(machineIDs, n_epochs)
    random_sensors = np.random.choice(sensors, n_epochs)

    # we intialize an array with that will later hold a sample of timetamps
    random_timestamps = np.random.choice(timestamps, n_epochs)
    
    for i in range(0, n_epochs):
        # take a slice of the dataframe that only contains the measures of one random machine
        df_s = df[df['machineID'] == random_machines[i]]
        
        # smooth the values of one random sensor, using our run_avg function
        smooth_values = run_avg(df_s[random_sensors[i]].values, com)
        
        # create a data frame with two columns: timestamp, and smoothed values
        df_smooth = pd.DataFrame(data={'timestamp': df_s['datetime'].values, 'value': smooth_values})

        # load the results of batch AD for this machine and sensor
        anoms_s = anoms_batch[((anoms_batch['machineID'] == random_machines[i]) & (anoms_batch['errorID'] == random_sensors[i]))]
                
        # find the location of the t'th random timestamp in the data frame
        if np.random.random() < p_anoms:
            anoms_timestamps = anoms_s['datetime'].values
            np.random.shuffle(anoms_timestamps)
            counter = 0
            while anoms_timestamps[0] < timestamps[0]:
                if counter > 100:
                    return 0.0, 9999.0
                np.random.shuffle(anoms_timestamps)
                counter += 1
            random_timestamps[i] = anoms_timestamps[0]
            
        # select the test case
        test_case = df_smooth[df_smooth['timestamp'] == random_timestamps[i]]
        test_case_index = test_case.index.values[0]


        # check whether the batch AD found an anomaly at that time stamps and copy into y_true at idx
        y_true_i = random_timestamps[i] in anoms_s['datetime'].values

        # perform online AD, and write result to y_pred
        y_pred_i, run_times_i = detect_ts_online(df_smooth, window_size, test_case_index)
        
        y_true.append(y_true_i)
        y_pred.append(y_pred_i)
        run_times.append(run_times_i)
            
    return fbeta_score(y_true, y_pred, beta=2), np.mean(run_times)
    