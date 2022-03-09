import time
import os
import glob
import numpy as np
import pandas as pd
import features_extraction
import statistics

from machine_learnig import classifier
from data_reduction import pca_components
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import accuracy_score


def split_data(X_train, X_val, y_train, y_val):
    # Split Data to Train and Validation
    split_index = [-1]*len(X_train) + [0]*len(X_val)
    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)
    pds = PredefinedSplit(test_fold=split_index)
    return X, y, pds


def preprocessing(data_path, fs, window_size, window_step, lowpass_cutoff_freq, save_csv_path):

    target_df = pd.read_csv(r'C:\\Git\\Motor Fault Detection\\Datasets\\Target.txt', delimiter="\t", header=None)
    target_df.values.tolist()
    all_files = glob.glob(os.path.join(data_path, "*.csv"))
    dataframe_list = [pd.read_csv(filename,
                                  names=["Acc_Inner_X", "Acc_Inner_Y", "Acc_Inner_Z",
                                         "Acc_Outer_X", "Acc_Outer_Y", "Acc_Outerr_Z"],
                                  usecols=[1,2,3,4,5,6])
                      for filename in all_files]

    # limpar colunas inuteis
    dataframe_list = features_extraction.upsampling(dataframe_list)
    filtered_dataframe_list = [pd.DataFrame(features_extraction.lowpass_filter(dataframe_list[i], lowpass_cutoff_freq,fs),
                                            columns=dataframe_list[i].columns)
                               for i in range(len(dataframe_list))]

    # Extract Feature
    data = [features_extraction.feature_dataset(dataframe, fs, window_size, window_step)
            for dataframe in filtered_dataframe_list]

    target_data = [pd.Series(np.full((len(dataframe.index)), target_df.values.tolist()[0][index]).tolist())
                   for index, dataframe in enumerate(data)]

    for i in range(len(target_data)):
        data[i]['Target'] = target_data[i]

    # Generate Full Feature DataFrame
    feature_df = pd.concat(data, axis=0, sort=False).reset_index(drop=True)
    feature_df.to_csv(os.path.join(save_csv_path, r'Feature_DataFrame.csv'))

    # PCA
    # X_pca = pca_components(data=feature_df.iloc[:, :-1], component_percent=0.95, graph=True)
    Y = feature_df.iloc[:, -1].to_numpy()
    # del feature_df

    X_minMax = MinMaxScaler().fit_transform(feature_df.iloc[:, :-1].to_numpy())
    #print(X_minMax[0])
    #X_minMax = MinMaxScaler().fit_transform(X_pca)

    # Slip in Train: 70% - Val: 15% - Test: 15%
    X_train, X_aux, y_train, y_aux = train_test_split(X_minMax, Y, test_size=0.1, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_aux, y_aux, test_size=0.5, random_state=42)
    del X_aux, y_aux
    X_cv, y_cv, psd = split_data(X_train, X_val, y_train, y_val)

    return X_cv, X_test, y_cv, y_test, psd


if __name__ == '__main__':
    start = time.time()


    # CSV Files Paths
    path = r'C:\\Git\\Motor Fault Detection\\Datasets'
    save_csv_path = r'C:\\Git\\Motor Fault Detection\\Teste_Data\\saved_csv'

    X_cv, X_test, y_cv, y_test, psd = preprocessing(data_path=path, fs=50000, window_step=250, window_size=500,
                                                    lowpass_cutoff_freq=20000 , save_csv_path=save_csv_path)
    prepro_time=time.time()
    print('pre_pross_fining')
    print(f'time: {(prepro_time-start)/60} min')
    for _ in range(2):
        y_fill = []
        test_acc = []
        loop_start = time.time()
        for clf in ['Random Forest', 'XGBoost', 'Adaboost', 'Neural Network']:
            out = classifier(X_cv, y_cv, X_test, y_test, psd, clf, 'Genetic_Search')
            y_fill.append(out[-1])
        y_vec = [statistics.mode([y_fill[0][i], y_fill[1][i], y_fill[2][i]]) for i in range(len(y_test))]
        print(f'MetaClassifier Test Acc: {accuracy_score(y_test, y_vec)}')
        loop_stop = time.time()
        print(f'Metaclass total time: {(loop_stop-loop_start)/60}min')
    end = time.time()
    print('loop Elapsed Time: ', (end - start)/60)