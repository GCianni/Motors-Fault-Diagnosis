import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, entropy
from scipy.signal import butter, lfilter, freqz
from sklearn.utils import resample


def probability(sample):
    dist = norm(np.mean(sample), np.std(sample))
    return dist.pdf(sample)


def fft_features(signal, f_sample):
    n = len(signal) // 2 + 1
    fft = 2.0 * np.abs(np.fft.fft(signal)[:n]) / n
    freq = np.linspace(0, f_sample / 2, n, endpoint=True)
    mean = np.mean(fft)
    centroid = np.average(fft, weights=freq)
    max_val = np.max(fft)
    kurt = kurtosis(fft)
    return mean, centroid, max_val, kurt


def upsampling(dataframes):
    sample_number = [len(dataframes[i]) for i in range(len(dataframes))]
    max_sample = max(sample_number)
    for i in range(len(sample_number)):
        if sample_number[i]<max_sample:
            dataframes[i] = resample(dataframes[i], n_samples=max_sample)
            dataframes[i].reset_index(drop=True)
    return dataframes


def lowpass_filter(data,cutoff,fs,order=5):

    normal_cutoff = cutoff / (0.5 * fs)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data).tolist()
    return filtered_data


def feature_dataset(dataframe, fs, window_size, window_step,):
    last_index = len(dataframe) - 1
    df_appender = []
    for channel in (dataframe.columns.values.tolist()):
        windowed_data = extract_windows_vectorized(dataframe[channel].to_numpy(), 0,
                                                   last_index - window_step, window_size, window_step)
        feature_df = feature_extraction(windowed_data, fs)
        feature_df.rename({0: 'RMS ' + channel, 1: 'Kurtosis ' + channel, 2: 'Entropy ' + channel,
                           3: 'Spectral Mean ' + channel, 4: 'Spectral Centroid ' + channel,
                           5: 'Spectral Maximum ' + channel, 6: 'Spectral Kurtosis ' + channel},
                          axis=1,inplace=True)
        df_appender.append(feature_df)

    return pd.concat(df_appender, axis=1)


def extract_windows_vectorized(array, clearing_time_index, max_time, sub_window_size, step_size):
    # max_time must be lower than (total lenght - step)
    start = clearing_time_index + 1 - sub_window_size + 1
    sub_windows = (
            start +
            np.expand_dims(np.arange(sub_window_size), 0) +
            np.expand_dims(np.arange(max_time + 1, step=step_size), 0).T
    )
    return array[sub_windows]


def feature_extraction(array, fs):
    feature_dataframe = pd.DataFrame(
          [[np.sqrt(np.mean(window_data ** 2)), kurtosis(window_data), entropy(probability(window_data)),
           fft_features(window_data, fs)] for window_data in array])
    spectral_data = feature_dataframe[3]
    feature_dataframe.drop(labels=[3], axis=1, inplace=True)
    feature_dataframe[[3, 4, 5, 6]] = pd.DataFrame(spectral_data.tolist(), index=feature_dataframe.index)
    return feature_dataframe


def get_data(filepath, ):

    return data
