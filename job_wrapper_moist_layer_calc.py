import os
import numpy as np

times = np.arange(
        '2021-06-01T00:00:00', '2021-07-01T00:00:00', dtype='datetime64[3h]')
for time in times:
    time_str = str(time)
    os.system("sbatch "
            "/home/u/u300676/moist-layers/batch_moist_layer_calc.sh "
            f"{time}")
