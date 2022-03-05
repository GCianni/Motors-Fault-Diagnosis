import time
import pandas as pd
import features_extraction
import os
import glob
import numpy as np
from data_reduction import pca_components
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def preprocessing(data_path, fs, window_size, window_step, lowpass_cutoff_freq, save_csv_path):
    target_df = pd.read_csv(r'C:\\Git\\Motor Fault Detection\\Teste_Data\\Target.txt', delimiter="\t", header=None)
    target_df.values.tolist()
    all_files = glob.glob(os.path.join(data_path, "*.csv"))
    dataframe_list = [pd.read_csv(filename) for filename in all_files]

    # limpar colunas inuteis

    dataframe_list = features_extraction.upsampling(dataframe_list)
    filtered_dataframe_list = [pd.DataFrame(features_extraction.lowpass_filter(dataframe_list[i], lowpass_cutoff_freq, fs),
                                            columns=dataframe_list[i].columns) for i in range(len(dataframe_list))]

    # Extract Feature
    data = [features_extraction.feature_dataset(dataframe, fs, window_size, window_step) for dataframe in filtered_dataframe_list]

    target_data = [pd.Series(np.full((len(dataframe.index)), target_df.values.tolist()[0][index]).tolist())
                   for index, dataframe in enumerate(data)]

    for i in range(len(target_data)):
        data[i]['Target'] = target_data[i]

    # Generate Full Feature DataFrame
    feature_df = pd.concat(data, axis=0, sort=False).reset_index(drop=True)
    feature_df.to_csv(os.path.join(save_csv_path, r'Feature_DataFrame.csv'))

    #PCA
    X_pca = pca_components(data=feature_df.iloc[:, :-1], component_percent=0.99, graph=False)
    Y = feature_df.iloc[:, -1].to_numpy()
    del feature_df

    X_minMax = MinMaxScaler().fit_transform(X_pca)

    # Slip in Train: 70% - Val: 15% - Test: 15%
    X_train, X_aux, y_train, y_aux = train_test_split(X_minMax, Y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_aux, y_aux, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == '__main__':
    start = time.time()

    path = r'C:\\Git\\Motor Fault Detection\\Teste_Data'# CSV Files Path
    save_csv_path = r'C:\\Git\\Motor Fault Detection\\Teste_Data\\saved_csv'
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing(data_path=path, fs=10000, window_step=1, window_size=10,
                                                                   lowpass_cutoff_freq=3000, save_csv_path=save_csv_path)
    #print(X_train, y_train)
    end = time.time()
    print('loop Elapsed Time: ', end - start)

