# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 09:05:55 2019

@author: DELL
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#matplotlib.use("Qt5Agg")
from pyts.image import gaf
#from pyts.approximation import paa
from pyts.approximation.paa import PiecewiseAggregateApproximation
#from pyts.approximation import sax
#from pyts.approximation import dft
#from pyts.approximation import mcb
#from pyts.approximation import sfa
import time

time_start=time.time()

path = r'E:\寒假\代码相关\单变量时间序列分类_2019_12_20\Step_1_data_preprocess\equalized_data.csv'

# In[0] 设置path和superparameters
# set path 
path_data = r'E:\寒假\代码相关\单变量时间序列分类_2019_12_20\Step_1_data_preprocess\equalized_data.csv'

def data_dimention_reduction(path_data, window_size):
    df_high_D = pd.read_csv(open(path_data), index_col = None) 
#    df_high_D_series = df_high_D.iloc[:,:-1]
    df_high_D_label = df_high_D.iloc[:,-1]
    
    # In[1] Series dimensionality reduction 
    # Paa transformation
    window_size = 6
    high_D = 6000
    low_D = high_D//window_size
    
    # PAA transformation
    paa = PiecewiseAggregateApproximation
    paa_ = paa(window_size=window_size)
    array_low_D = paa_.fit_transform(df_high_D.values)  
    df_low_D_series = pd.DataFrame(array_low_D, columns = range(low_D+1),index = df_high_D.index)
    
    #print(df_1000_series)
    df_low_D_label = df_high_D_label
    df_low_D = pd.concat([df_low_D_series,df_low_D_label], axis = 1)
    #print(df_1000) 
    return df_low_D

df_low_D = data_dimention_reduction(path_data, window_size = 6)

# In[0] data generateion
def data_load_xcc(df_data,proportion):
    '''
    给分类器提供不同比例的训练集和测试集
    '''
#    paht_data = r'X:\Work of Lab\单变量时间序列分类_2019_12_20\Step_1_data_preprocess\data_normalized.csv'
    df_normalized = df_data
    
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

X,y,X_,y_ = data_load_xcc(df_low_D, 0.7)

# In[3] transform the time series into GAF
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image.gaf import GramianAngularField

gasf = GramianAngularField(image_size = 48, method = 'summation')
X_gasf = gasf.fit_transform(X)
gadf = GramianAngularField(image_size = 48, method = 'difference')
X_gadf  = gadf.fit_transform(X)

# In[2] Show the images for the first time series
fig = plt.figure(figsize=(8,4))
grid = ImageGrid(fig, 111, 
                 nrows_ncols=(1,2), 
                 axes_pad = 0.15,
                 share_all = True, 
                 cbar_location = 'right',
                 cbar_mode = 'single',
                 cbar_size = '7%',
                 cbar_pad = 0.3,
                 )

images = [X_gasf[201], X_gadf[201]]
titles = ['Summation','Difference']
for image, title, ax in zip(images, titles, grid):
    im = ax.imshow(image, cmap = 'rainbow', origin = 'lower')
    ax.set_title(title, fontdict = {'fontsize':12})
#    im.savefig('1.jpg')
#    ax.saveGrid('image.jpg')
    
ax.cax.colorbar(im) 
ax.cax.toggle_label(True)
plt.suptitle('gramian Angular Fields', y=0.98, fontsize=16)
plt.show()
plt.savefig('image.jpg')


