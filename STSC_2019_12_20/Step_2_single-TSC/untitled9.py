# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 23:56:13 2019

SVM original code

@author: DELL
"""

import pandas 
import keras
import sklearn 
from sklearn import svm
import matplotlib.pyplot as plt

clf = svm.NuSVC(gamma = 'auto')
clf.fit(X,Y)

# plot the decision for each datapoint on the grid
Z = clf.decision_function(np.)

 
 