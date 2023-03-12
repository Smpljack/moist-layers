import os
import numpy as np

times = np.arange('2021-07-28', '2021-08-11', dtype='datetime64[D]')
for time in times:
    time_start = str(time)
    time_end = str(time + np.timedelta64(1, 'D'))
    os.system("sbatch "
            "/home/u/u300676/moist-layers/batch_map_4d_eml_labels.sh "
            f"{time_start} {time_end}")