# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 13:24:32 2019

kNN & SVAVSM & BOSSVS

@author: DELL

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyts.classification import KNeighborsClassifier
from pyts.classification import SAXVSM 
from pyts.classification import BOSSVS

path_result = '/content/drive/My Drive/STSC_2019_12_20/Step_2_single-TSC/kNN_SAX_BOSSVS'



def data_generation(path,proportion):
    
    df = pd.read_csv(open(path))
    
    #df_features = pd.DataFrame(columns = ['mean_seq','sigma_seq','skewness_3_seq',
    #                                      'skewness_4_seq','slope_seq','wave_freq_seq','label_seq'])
    
    
    # key number
    split_point = int(proportion*df.shape[0])
    
    # 逐列归一化
    df_test = df
    #df_test = (df_test - df_test.min()) / (df_test.max() - df_test.min())
    #df_test['label_seq'] =df_test['label_seq'].map(lambda x:int(x*9))
    #print(df_test)
    
    #_______________________________ Data split
    
    train_feature = df_test.iloc[:split_point,:-1]
#    train_feature.to_csv('train_feature.csv',index=None,columns=None)
    print(train_feature.shape)
    
    train_label = df_test.iloc[:split_point,-1]  
#    train_label.to_csv('train_label.csv',index=None)
    print(train_label.shape)
    
    test_feature = df_test.iloc[split_point:,:-1]
#    test_feature.to_csv('test_feature.csv',index=None,columns=None)
    print(test_feature.shape)
    
    test_label = df_test.iloc[split_point:,-1] 
#    test_label.to_csv('test_label.csv',index=None)
    print(test_label.shape)
    
    return train_feature, test_feature, train_label, test_label

# In[3] SVAVSM method

def main_SVAVSM(proportion,word_size):

    path = '/content/drive/My Drive/STSC_2019_12_20/Step_1_data_preprocess/equalized_data.csv'
    X_train, X_test, y_train, y_test = data_generation(path,proportion)
    
    clf = SAXVSM(window_size=4, sublinear_tf=False, use_idf=False)
    
    clf.fit(X_train, y_train)
    
    score = clf.score(X_test, y_test)
    
    return score

#print(main_SVAVSM(0.7, 26))

def best_proportion_and_window_size_for_SVAVSM():
    word_size = 2
    record_df = pd.DataFrame(index=range(12,18),columns=range(word_size,27,4))
    for i in range(12,18):
        proportion = i/20
        data_record_window_size = []
        for j in range(word_size,27,4):
            window_size = j
            score = main_SVAVSM(proportion, word_size)
            data_record_window_size.append(score)
        row_record = pd.DataFrame(data_record_window_size).T
        row_record.columns = record_df.columns
        record_df = pd.concat([record_df,row_record])  
    #plt.bar(range(len(record_df)), record_df, label = 'record_df')
    record_df.to_csv(path_result + '/best_proportion_and_wondow_size_for_SVAVSM.csv')
    return record_df

best_proportion_and_window_size_for_SVAVSM()


