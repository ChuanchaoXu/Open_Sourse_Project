# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 13:43:33 2019
Random Forest 
@author: XI
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# In[0] Data split
'''
# 分为train_feature、train_label、test_feature和test_label四个文件供后续程序使用
'''
def data_load_xcc(path_data, proportion):
    '''
    给分类器提供不同比例的训练集和测试集
    '''
#    paht_data = '/content/drive/My Drive/STSC_2019_12_20\Step_1_data_preprocess\data_normalized.csv'
    df_normalized = pd.read_csv(open(path_data))
    
    split_point = int(proportion*df_normalized.shape[0])
    
    train_series = df_normalized.iloc[:split_point,:-1]
    train_labels = df_normalized.iloc[:split_point,-1]
    
    test_series = df_normalized.iloc[split_point:,:-1]
    test_labels = df_normalized.iloc[split_point:,-1]
    
    X = train_series.values
    y = train_labels.values.flatten().astype('int')
    X_ = test_series.values
    y_ = test_labels.values.flatten().astype('int')
    
    return X,y,X_,y_

def data_load_xcc_original_train_test(df_train,df_test):
    X = df_train.iloc[:,:-1].values
    y = df_train.iloc[:,-1].values
    X_ = df_train.iloc[:,:-1].values
    y_ = df_train.iloc[:,-1].values
    
    return X,y,X_,y_

# In[example 2]
import sklearn as skl   # machine learning
from sklearn.ensemble import RandomForestClassifier 
#from plotnine import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


# In[4]
# Three methods

from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm  

def methods_classifier(X,y,X_,y_):
    
    #_______________DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
    clf.fit(X, y)
    #print(clf.feature_importances_)
    t = 0
    for i in range(len(X_)):
        input = X_[i].tolist()
        output = clf.predict([input])
    #    print('predict:',output,'real:',y_[i])
        if output == y_[i]:
            t += 1
    acc_1 = round(t/len(X_),3)
    print('决策树Acc: %.2f%%' % (acc_1*100))
    #scores = cross_val_score(clf, X, y)
    #print(scores)
    #print(scores.mean())   

                          
    #_______________RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
    clf.fit(X, y)
    #print(clf.feature_importances_)
    t = 0
    for i in range(len(X_)):
        input = X_[i].tolist()
        output = clf.predict([input])
    #    print('predict:',output,'real:',y_[i])
        if output == y_[i]:
            t += 1
    acc_2 = round(t/len(X_),3)
    print('随机森林Acc: %.2f%%' % (acc_2*100))
    #scores = cross_val_score(clf, X, y)
    #print(scores)
    #print(scores.mean())    
    
    
    #_______________ExtraTreesClassifier
    clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
    clf.fit(X, y)
    #print(clf.feature_importances_)
    t = 0
    for i in range(len(X_)):
        input = X_[i].tolist()
        output = clf.predict([input])
    #    print('predict:',output,'real:',y_[i])
        if output == y_[i]:
            t += 1
    acc_3 = round(t/len(X_),3)
    print('极端随机树Acc: %.2f%%' % (acc_3*100))
    #scores = cross_val_score(clf, X, y)
    #print(scores)
    #print(scores.mean())
    
    #______________SVM
    clf = svm.SVC(kernel='rbf')  
    clf.fit(X, y)
    #print(clf.feature_importances_)
    t = 0
    for i in range(len(X_)):
        input = X_[i].tolist()
        output = clf.predict([input])
    #    print('predict:',output,'real:',y_[i])
        if output == y_[i]:
            t += 1
    acc_4 = round(t/len(X_),3)
    print('支持向量机Acc: %.2f%%' % (acc_4*100))
    
    #scores = cross_val_score(clf, X, y)
    #print(scores)
    #print(scores.mean())
    
    return acc_1,acc_2,acc_3,acc_4


def classifier(i,j):
    method_1_acc = []
    method_2_acc = []
    method_3_acc = []
    method_4_acc = []
    k_for_split = []
    
    record_df = pd.DataFrame(index=['决策树Acc','随机森林Acc', '极端随机数Acc','支持向量机',None])
    path_content= '/content/drive/My Drive/UTC&MTC/RUL-Net-master/XCC_File/multi_class'
    path_data = path_content + '/'+j+'/sample_'+str(i)+'.csv'
    
    all_record_df = pd.DataFrame(columns = [str(float(m/20)) for m in range(12,20)])
    
    path_D_R_E_S = '/content/drive/My Drive/STSC_2019_12_20/Step_2_single-TSC/D_R_E_S'
    
    for n in range(12,20):
        k = float(n/20)
        k_for_split.append(k)
        print('\nsplit_point:',k)
        X,y,X_,y_ = data_load_xcc(path_data,k)
        acc_1,acc_2,acc_3,acc_4 = methods_classifier(X,y,X_,y_)
        
        record_df[str(k)] = [acc_1, acc_2, acc_3, acc_4, None]
        method_1_acc.append(acc_1)
        method_2_acc.append(acc_2)
        method_3_acc.append(acc_3)
        method_4_acc.append(acc_4)
    
    
    
    max_acc_1 = max(method_1_acc)
    index_1 = method_1_acc.index(max_acc_1)
    k_1 = k_for_split[index_1]
    
    max_acc_2 = max(method_2_acc)
    index_2 = method_2_acc.index(max_acc_2)
    k_2 = k_for_split[index_2]
    
    max_acc_3 = max(method_3_acc)
    index_3 = method_3_acc.index(max_acc_3)
    k_3 = k_for_split[index_3]
    
    max_acc_4 = max(method_4_acc)
    index_4 = method_4_acc.index(max_acc_4)
    k_4 = k_for_split[index_4]
        
        


    print('DecisionTree_Classifier:',max_acc_1,k_1,'\n',
          'RandomForest_Classifier:',max_acc_2,k_2,'\n',
          'ExtraTrees_Classifier:',max_acc_3,k_3,'\n',
          'SVM_Classifiier:',max_acc_4,k_4
          )
        
#        record_df.to_excel(path_D_R_E_S + '/D_R_E_S_tf_' + path_data[-5] + '.xlsx')
    all_record_df = pd.concat([all_record_df,record_df])
#        if int(path_data[-5]) == 7:
#            break
    

    all_record_df.to_csv(path_content + '/'+j+'/Acc_'+str(i)+'.csv')
 
 
def main():
    
    sample_type = {'eq':[127,118,109,105],'uneq':[136,190,280]}
    #随机生成均衡/不均衡样本
    for j in sample_type.keys():
      for i in sample_type[j]:
          classifier(i,j)
          print('time')

#main()


def RF_Log_Svm_cv():
    k = 0.7
    path_data = '/content/drive/My Drive/STSC_2019_12_20/Step_1_data_preprocess/data_normalized.csv'
    X,y,X_,y_ = data_load_xcc(path_data,k)
    data_input,data_output = X,y
        
    from sklearn.ensemble import RandomForestClassifier   # 随即森林模型
    from sklearn.linear_model import LogisticRegression   # 逻辑回归模型
    from sklearn import svm     # 支持向量机
    from sklearn.model_selection import cross_val_score
     
    # 模型重命名   
    rf_class = RandomForestClassifier(n_estimators=10) 
    log_class = LogisticRegression()
    svm_class = svm.LinearSVC()
    
    # 把数据分为四分，并计算每次交叉验证的结果，并返回
    print(cross_val_score(rf_class, data_input, data_output, scoring='accuracy', cv = 4))
    
    # 这里的cross_val_score将交叉验证的整个过程连接起来，不用再进行手动的分割数据
    # cv参数用于规定将原始数据分成多少份
    accuracy = cross_val_score(rf_class, data_input, data_output, scoring='accuracy', cv = 4).mean() * 100
    print("Accuracy of Random Forests is: " , accuracy)
    
    accuracy = cross_val_score(log_class, data_input, data_output, scoring='accuracy', cv = 4).mean() * 100
    print("Accuracy of logistic is: " , accuracy)
    
    accuracy = cross_val_score(svm_class, data_input, data_output, scoring='accuracy', cv = 4).mean() * 100
    print("Accuracy of SVM is: " , accuracy)
    

#RF_Log_Svm_cv()









