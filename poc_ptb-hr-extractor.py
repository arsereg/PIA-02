# 'data/ptb-diagnostic-ecg-database-1.0.0/patient002/s0015lre'

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# Step 1: Load the ECG Data
# Replace 'record_name' with the actual name of the record file you're using
record = wfdb.rdrecord('data/ptb-diagnostic-ecg-database-1.0.0/patient002/s0015lre')


# Step 3: Bandpass Filter to Remove Noise
def bandpass_filter(signal, lowcut, highcut, fs, order=1):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

ecg_data = record.p_signal  # Load all 12-lead ECG signals

# Step 2: Select Lead II for Heart Rate Calculation
lead_II = ecg_data[:, 1]  # Assuming lead II is the second column

fs = 1000  # Sampling frequency is 1000 Hz
filtered_lead_II = bandpass_filter(lead_II, 0.5, 50, fs)

# Step 4: R-Peak Detection
# Use the find_peaks function to detect R-peaks
peaks, _ = find_peaks(filtered_lead_II, distance=fs*0.6)  # Assuming a minimum distance of 600ms between peaks

# Step 5: Calculate RR Intervals and Heart Rate
rr_intervals = np.diff(peaks) / fs  # RR intervals in seconds
hr_values = 60 / rr_intervals  # Heart rate in beats per minute (bpm)

# Step 6: Generate Time Axis for Heart Rate Plot
time_peaks = peaks / fs  # Time of R-peaks in seconds
time_hr = (time_peaks[:-1] + time_peaks[1:]) / 2  # Midpoint between successive peaks

# Step 7: Visualize the Heart Rate
plt.figure(figsize=(10, 6))
plt.plot(time_hr, hr_values, label='Heart Rate (bpm)', color='b', marker='o', linestyle='-')
plt.title('Heart Rate over Time')
plt.xlabel('Time (s)')
plt.ylabel('Heart Rate (bpm)')
plt.grid(True)
plt.legend()
plt.show()
