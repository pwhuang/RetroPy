import numpy as np

def convergence_rate(err_norm, step_size):
    return np.diff(np.log(err_norm))/np.diff(np.log(step_size))
