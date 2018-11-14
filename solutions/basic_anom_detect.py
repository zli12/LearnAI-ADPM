
def detect_anomalies(ts, tolerance=4):
    m = np.mean(ts)
    std = np.std(ts)

    # find the array indices of extreme values (anomalies)
    idx = np.where(ts > m + tolerance * std)[0].tolist()

    # create an array that is all NaN, except for the anomalies
    anoms = np.full(ts.shape[0], np.nan)
    anoms[idx] = ts[idx] # copy the value of the anomaly
    
    return anoms