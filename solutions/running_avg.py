def run_avg(ts):
    rm_o = np.zeros_like(ts)
    rm_o[0] = ts[0]
    
    for r in range(1, len(ts)):
        curr_com = float(min(com, r))
        rm_o[r] = rm_o[r-1] + (ts[r] - rm_o[r-1])/(curr_com + 1)
    
    return rm_o