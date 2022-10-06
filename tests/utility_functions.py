# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import matplotlib.pyplot as plt
from dolfin.common.plotting import mplot_function

def convergence_rate(err_norm, step_size):
    return np.diff(np.log(err_norm))/np.diff(np.log(step_size))

def quick_plot(func1, func2):
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    mplot_function(ax[0], func1)
    mplot_function(ax[1], func2)
    plt.show()
