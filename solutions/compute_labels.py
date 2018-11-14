for i in range(1, 5): # iterate over the four components
    # find all the times a component failed for a given machine
    df_temp = df_all.loc[df_all['f_' + str(i)] == 1, ['machineID', 'datetime']]
    label = 'y_' + str(i) # name of target column (one per component)
    df_all[label] = 0
    for n in range(df_temp.shape[0]): # iterate over all the failure times
        machineID, datetime = df_temp.iloc[n, :]
        dt_end = datetime - pd.Timedelta('3 hours') # 3 hours prior to failure
        dt_start = datetime - pd.Timedelta('2 days') # n days prior to failure
        if n % 500 == 0: 
            print("a failure occured on machine {0} at {1}, so {2} is set to 1 between {4} and {3}".format(machineID, datetime, label, dt_end, dt_start))
        df_all.loc[(df_all['machineID'] == machineID) & 
                   (df_all['datetime'].between(dt_start, dt_end)), label] = 1