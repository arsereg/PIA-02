import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
import antropy as ant


def calcular_min_hrv(hr):
    min_hrv = 1000
    previous_hr = 0
    for i in range(len(hr)):
        if i == 0:
            previous_hr = hr[i]
        else:
            hrv = abs(hr[i] - previous_hr)
            if hrv != 0 and hrv < min_hrv:
                min_hrv = hrv
            previous_hr = hr[i]
    return min_hrv


def calcular_max_hrv(hr):
    max_hrv = 0
    previous_hr = 0
    for i in range(len(hr)):
        if i == 0:
            previous_hr = hr[i]
        else:
            hrv = abs(hr[i] - previous_hr)
            if hrv > max_hrv:
                max_hrv = hrv
            previous_hr = hr[i]
    return max_hrv


def calcular_mean_hrv(hr):
    mean_hrv = 0
    previous_hr = 0
    for i in range(len(hr)):
        if i == 0:
            previous_hr = hr[i]
        else:
            hrv = abs(hr[i] - previous_hr)
            mean_hrv += hrv
            previous_hr = hr[i]
    return mean_hrv / len(hr)


def calcular_median_hrv(hr):
    hrvs = []
    previous_hr = 0
    for i in range(len(hr)):
        if i == 0:
            previous_hr = hr[i]
        else:
            hrv = abs(hr[i] - previous_hr)
            hrvs.append(hrv)
            previous_hr = hr[i]
    return np.median(hrvs)


def calcular_std_hrv(hr):
    hrvs = []
    previous_hr = 0
    for i in range(len(hr)):
        if i == 0:
            previous_hr = hr[i]
        else:
            hrv = abs(hr[i] - previous_hr)
            hrvs.append(hrv)
            previous_hr = hr[i]
    return np.std(hrvs)


def count_outliers(hr):
    q1 = np.percentile(hr, 5)
    q3 = np.percentile(hr, 95)
    outliers = 0
    for value in hr:
        if value < q1 or value > q3:
            outliers += 1
    return outliers


def get_features(hr_values, time_hr, record_name, diagnosis):
    highest_hr = max(hr_values)
    lowest_hr = min(hr_values)
    mean_hr = np.mean(hr_values)
    median_hr = np.median(hr_values)
    std_hr = np.std(hr_values)
    min_hrv = calcular_min_hrv(hr_values)
    max_hrv = calcular_max_hrv(hr_values)
    mean_hrv = calcular_mean_hrv(hr_values)
    median_hrv = calcular_median_hrv(hr_values)
    outliers = count_outliers(hr_values)
    std_hrv = calcular_std_hrv(hr_values)

    # HR Slope: Representa cambios repentinos en la frecuencia cardíaca
    delta_time = np.diff(time_hr)
    delta_hr = np.diff(hr_values)
    hr_slope = delta_hr / delta_time
    mean_hr_slope = np.mean(hr_slope)
    max_hr_slope = np.max(hr_slope)
    std_hr_slope = np.std(hr_slope)

    # HR Variability Frequency Domain Analisis con FFT (Fast Fourier Transform)
    hrv = np.diff(hr_values)
    N = len(hrv)
    sampling_rate = 1 / np.mean(np.diff(time_hr))
    fft_values = fft(hrv)
    frequencies = fftfreq(N, d=1 / sampling_rate)

    # Se consideran solo frecuencias positivas
    positive_frequencies = frequencies[:N // 2]
    positive_fft_values = np.abs(fft_values[:N // 2])  # Magnitud de frecuencias positivas

    vlf_power = np.sum(positive_fft_values[(positive_frequencies >= 0.003) & (positive_frequencies < 0.04)])
    lf_power = np.sum(positive_fft_values[(positive_frequencies >= 0.04) & (positive_frequencies < 0.15)])
    hlf_power = np.sum(positive_fft_values[(positive_frequencies >= 0.15) & (positive_frequencies < 0.4)])

    # Analisis de RR Interval

    peaks, _ = find_peaks(hr_values, distance=1)
    peak_times = time_hr[peaks]
    rr_intervals = np.diff(peak_times)

    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals)
    min_rr = np.min(rr_intervals)
    max_rr = np.max(rr_intervals)

    # Tendencia de la frecuencia cardiaca

    window_size = 10
    hr_smoothed = pd.Series(hr_values).rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(
     method='ffill').values

    time_indices = np.arange(len(hr_smoothed)).reshape(-1, 1)

    linear_model = LinearRegression()
    linear_model.fit(time_indices, hr_smoothed)

    hr_trend_line = linear_model.predict(time_indices)
    trend_slope = linear_model.coef_[0]
    deviation_from_trend = hr_smoothed - hr_trend_line

    mean_deviation = np.mean(deviation_from_trend)
    std_deviation = np.std(deviation_from_trend)

    # Mediciones de Entropía

    approximation_entropy = ant.app_entropy(hr_values, order=2, metric='euclidean')
    sample_entropy = ant.sample_entropy(hr_values, order=2, metric='euclidean')

    result = {
        'patient': record_name,
        'diagnosis': diagnosis,
        'highest_heart_rate': highest_hr,
        'lowest_heart_rate': lowest_hr,
        'mean_heart_rate': mean_hr,
        'standard_deviation_hr': std_hr,
        'minimum_hrv': min_hrv,
        'maximum_hrv': max_hrv,
        'mean_hrv': mean_hrv,
        'median_hrv': median_hrv,
        'standard_deviation_hrv': std_hrv,
        'mean_hr_slope': mean_hr_slope,
        'Max_hr_slope': max_hr_slope,
        'standard_deviation_hr_slope': std_hr_slope,
        'vlf_power': vlf_power,
        'lf_power': lf_power,
        'hf_power': hlf_power,
        'mean_rr': mean_rr,
        'standard_deviation_rr': std_rr,
        'minimum_rr': min_rr,
        'maximum_rr': max_rr,
        'mean_deviation': mean_deviation,
        'tendency_standard_deviation': std_deviation,
        'approximation_entropy': approximation_entropy,
        'sample_entropy': sample_entropy,
        'outliers': outliers
    }

    return result