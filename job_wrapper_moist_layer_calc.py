import os
import numpy as np

times = np.arange(
        '2021-01-01', '2021-02-01', dtype='datetime64[1D]')
for time in times:
    time_str = str(time)
    os.system("sbatch "
            "/home/u/u300676/moist-layers/batch_moist_layer_calc.sh "
            f"{time}")
