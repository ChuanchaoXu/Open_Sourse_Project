# -*- coding: utf-8 -*-

"""

Created on Thu Mar  5 20:31:11 2020

@author: XCL
"""
#pip install pyts

path = '/content/drive/My Drive/STSC_2019_12_20/Step_0_preparation/original_data.csv'
import pandas as pd
import numpy as np
from pyts.approximation import PiecewiseAggregateApproximation

window_size = 5
paa = PiecewiseAggregateApproximation(window_size=window_size)
file = pd.read_csv(path)

sample_ = file.iloc[12][50:550]
sample = sample_.to_list()

time_mark_xcc = range(3, 500,window_size)

#sample_PAA = pd.DataFrame(paa.fit_transform(sample_.values))

sample_xcc = [np.mean(sample[i:i+window_size]) for i in range(1,int(505/window_size))]
import matplotlib.pyplot as plt
fig_1 = plt.figure() 
plt.plot(range(len(sample)),sample,color = 'blue',label='原始数据', linewidth=0.8)
plt.legend()
plt.plot(time_mark_xcc, sample_xcc, color = 'red',label='近似数据', linewidth=0.8)
plt.legend()

plt.show()