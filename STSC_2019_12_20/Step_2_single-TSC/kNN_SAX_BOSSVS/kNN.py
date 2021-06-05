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

# In[2] kNN method

def main_kNN(proportion):

    path = '/content/drive/My Drive/STSC_2019_12_20/Step_1_data_preprocess/equalized_data.csv'
    X_train, X_test, y_train, y_test = data_generation(path,proportion)
    
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return score

def best_proportion_for_kNN():
    data_record = []
    for i in range(5,18):
        proportion = i/20
        score = main_kNN(proportion)
        data_record.append(score)
    plt.bar(range(5,18), data_record, label = 'data_record')
    plt.savefig(path_result+'/kNN.svg')
    plt.show()

best_proportion_for_kNN()

