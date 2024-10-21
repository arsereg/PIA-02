import wfdb.plot
import wfdb.processing
import numpy as np
import wfdb
from wfdb import processing, rdrecord, rdann
data_location = 'data/mit-bih-arrhythmia-database-1.0.0/'
file = '200'
file_name = f'{data_location}/{file}'

record = wfdb.rdrecord(file_name, sampto=10000)
annotation = wfdb.rdann(file_name, 'atr', sampto=10000)
fs = record.fs
r_peaks = annotation.sample
rr_intervals = np.diff(r_peaks) / fs
hr_values = 60 / rr_intervals
print("Heart rate values (in bpm):", hr_values)

import matplotlib.pyplot as plt

time = r_peaks[1:] / fs  # Time in seconds for each HR measurement

plt.plot(time, hr_values)
plt.xlabel("Time (s)")
plt.ylabel("Heart Rate (bpm)")
plt.title("Heart Rate over Time")
plt.show()
print("Sampling frequency (fs):", record.fs)
