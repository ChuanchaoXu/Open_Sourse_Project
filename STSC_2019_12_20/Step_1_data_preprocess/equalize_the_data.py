# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 16:16:32 2019

@author: DELL
"""

import pandas as pd
import random
import numpy as np
from random import shuffle

# In[0]  equalize the data
path_0 = r'X:\Work of Lab\单变量时间序列分类_2019_12_20\Step_0_preparation\original_data.csv'
data_0 = pd.read_csv(open(path_0))

No_0 = []
No_1 = []
No_2 = []
No_3 = []
No_4 = []
No_5 = []
No_6 = []
No_7 = []
No_8 = []
No_9 = []

for i in range(data_0.shape[0]):
    if data_0.iloc[i,-1] == 0:
        No_0.append(i)
        
for i in range(data_0.shape[0]):
    if data_0.iloc[i,-1] == 1:
        No_1.append(i)
        
for i in range(data_0.shape[0]):
    if data_0.iloc[i,-1] == 2:
        No_2.append(i)

for i in range(data_0.shape[0]):
    if data_0.iloc[i,-1] == 3:
        No_3.append(i)
        
for i in range(data_0.shape[0]):
    if data_0.iloc[i,-1] == 4:
        No_4.append(i)
        
for i in range(data_0.shape[0]):
    if data_0.iloc[i,-1] == 5:
        No_5.append(i)
        
for i in range(data_0.shape[0]):
    if data_0.iloc[i,-1] == 6:
        No_6.append(i)
        
for i in range(data_0.shape[0]):
    if data_0.iloc[i,-1] == 7:
        No_7.append(i)
        
for i in range(data_0.shape[0]):
    if data_0.iloc[i,-1] == 8:
        No_8.append(i)
        
for i in range(data_0.shape[0]):
    if data_0.iloc[i,-1] == 9:
        No_9.append(i)
        

No_0_sample = random.sample(No_0, 47)
No_7_sample = random.sample(No_7, 47)
No_9_sample = random.sample(No_9, 47)

No_0_to_9 = No_0_sample + No_1 + No_2 + No_3 + No_4 + No_5 + No_6 + No_7_sample + No_8 + No_9_sample
No_0_to_9.sort()


data_1 = pd.DataFrame(columns = data_0.columns)
for i in No_0_to_9:
    data_1 = data_1.append(data_0.iloc[i,:])

data_1 = data_1.drop(list(data_1.columns)[0],axis = 1)
print(data_1.shape[0])
path_1 = r'X:\Work of Lab\单变量时间序列分类_2019_12_20\Step_1_data_preprocess\equalized_data.csv'
data_1.to_csv(path_1, index = None, header = None)
 
# In[1] devide the data
proportion = 0.7
split_point = int(proportion*data_1.shape[0])
index_data_1 = list(data_1.index)
shuffle(index_data_1)
No_train = [ i for i in index_data_1[:split_point]]
No_test = [ i for i in index_data_1[split_point:]]
 

## train data
data_train = pd.DataFrame(columns = data_1.columns)
for i in No_train:
    data_train = data_train.append(data_1.loc[i,:])
#    data_train = data_train.append(data_1.iloc[i,:])
print(data_train.shape[0])
path_train = r'X:\Work of Lab\单变量时间序列分类_2019_12_20\Step_1_data_preprocess\train.csv'
data_train.to_csv(path_train, index = None, header = None)

path_train_series = r'X:\Work of Lab\单变量时间序列分类_2019_12_20\Step_1_data_preprocess\train_series.csv'
data_train.iloc[:,:-1].to_csv(path_train_series, index = None, header = None)
path_train_labels = r'X:\Work of Lab\单变量时间序列分类_2019_12_20\Step_1_data_preprocess\train_labels.csv'
data_train.iloc[:,-1].to_csv(path_train_labels, index = None, header = None)
data_train.iloc[:,-1].to_txt('X:\Work of Lab\单变量时间序列分类_2019_12_20\Step_1_data_preprocess\train_series.txt')


## test data
data_test = pd.DataFrame(columns = data_1.columns)
for i in No_test:
    data_test =  data_test.append(data_1.loc[i,:])
print(data_test.shape[0])
path_test = r'X:\Work of Lab\单变量时间序列分类_2019_12_20\Step_1_data_preprocess\test.csv'
data_test.to_csv(path_test, index = None, header = None)

path_test_series = r'X:\Work of Lab\单变量时间序列分类_2019_12_20\Step_1_data_preprocess\test_series.csv'
data_test.iloc[:,:-1].to_csv(path_test_series, index = None, header = None)
path_test_labels = r'X:\Work of Lab\单变量时间序列分类_2019_12_20\Step_1_data_preprocess\test_labels.csv'
data_test.iloc[:,-1].to_csv(path_test_labels, index = None, header = None)











