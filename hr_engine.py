import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import antropy as ant




def calcular_min_hrv(hr):
    """
    Calcula la mínima variabilidad de la frecuencia cardiaca
    :param hr: Lista de valores de frecuencia cardiaca
    :return: Mínima variabilidad de la frecuencia cardiaca
    """
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
    """
    Calcula la máxima variabilidad de la frecuencia cardiaca
    :param hr: Lista de valores de frecuencia cardiaca
    :return: Máxima variabilidad de la frecuencia cardiaca
    """
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


def calcular_std_hrv(hr):
    """
    Calcula la desviación estándar de la variabilidad de la frecuencia cardiaca
    :param hr: Lista de valores de frecuencia cardiaca
    :return: Desviación estándar de la variabilidad de la frecuencia cardiaca
    """
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


def get_features(hr_values, time_hr, record_name, diagnosis):
    lowest_hr = min(hr_values)
    min_hrv = calcular_min_hrv(hr_values)
    max_hrv = calcular_max_hrv(hr_values)
    std_hrv = calcular_std_hrv(hr_values)

    # HR Slope: Representa cambios repentinos en la frecuencia cardíaca
    delta_time = np.diff(time_hr)
    delta_hr = np.diff(hr_values)
    hr_slope = delta_hr / delta_time
    mean_hr_slope = np.mean(hr_slope)

    # HR Variability Frequency Domain Analisis con FFT (Fast Fourier Transform)
    hrv = np.diff(hr_values)
    N = len(hrv)
    sampling_rate = 1 / np.mean(np.diff(time_hr))
    fft_values = fft(hrv)
    frequencies = fftfreq(N, d=1 / sampling_rate)

    # Se consideran solo frecuencias positivas
    positive_frequencies = frequencies[:N // 2]
    positive_fft_values = np.abs(fft_values[:N // 2])  # Magnitud de frecuencias positivas

    # Se obtienen los picos de la FFT
    vlf_power = np.sum(positive_fft_values[(positive_frequencies >= 0.003) & (positive_frequencies < 0.04)])
    lf_power = np.sum(positive_fft_values[(positive_frequencies >= 0.04) & (positive_frequencies < 0.15)])
    hlf_power = np.sum(positive_fft_values[(positive_frequencies >= 0.15) & (positive_frequencies < 0.4)])


    # Tendencia de la frecuencia cardiaca

    window_size = 10
    hr_smoothed = pd.Series(hr_values).rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(
     method='ffill').values

    time_indices = np.arange(len(hr_smoothed)).reshape(-1, 1)

    linear_model = LinearRegression()
    linear_model.fit(time_indices, hr_smoothed)

    trend_slope = linear_model.coef_[0]


    # Mediciones de Entropía

    approximation_entropy = ant.app_entropy(hr_values, order=2, metric='euclidean')

    result = {
        'diagnosis': diagnosis,
        'minimum_hrv': min_hrv,
        'maximum_hrv': max_hrv,
        'standard_deviation_hrv': std_hrv,
        'mean_hr_slope': mean_hr_slope,
        'tendency_slope': trend_slope,
        'lowest_heart_rate': lowest_hr,
        'vlf_power': vlf_power,
        'lf_power': lf_power,
        'hf_power': hlf_power,
        'approximation_entropy': approximation_entropy,
    }

    result = pd.DataFrame(result, index=[0])

    columns_to_reduce = [
        'vlf_power',
        'lf_power',
        'hf_power',
        'lowest_heart_rate',
    ]




    result[columns_to_reduce] = result[columns_to_reduce].apply(lambda x: x/100)

    important_features = [
        'minimum_hrv',
        'maximum_hrv',
        'standard_deviation_hrv',
        'mean_hr_slope',
        'tendency_slope',
        'lowest_heart_rate',
        'vlf_power',
        'lf_power',
        'hf_power',
        'approximation_entropy'
    ]

    features = result[important_features]
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    imputer = SimpleImputer(strategy='median')
    # scaler = StandardScaler()
    features = imputer.fit_transform(features)
    # features = scaler.fit_transform(features)
    df = pd.DataFrame(features, columns=important_features)
    df['diagnosis'] = result['diagnosis']
    return df.iloc[0].to_dict()


def scale_for_training(df):
    from sklearn.preprocessing import StandardScaler
    import joblib
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    df = imputer.fit_transform(df)
    df = scaler.fit_transform(df)
    joblib.dump(scaler, 'scaler.pkl')
    return df

def scale_for_prediction(df):
    import joblib
    scaler = joblib.load('scaler.pkl')
    df = scaler.transform(df)
    return df