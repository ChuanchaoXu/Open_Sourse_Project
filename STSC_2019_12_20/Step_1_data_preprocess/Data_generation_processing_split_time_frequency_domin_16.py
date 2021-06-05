# 时频域统计特征汇总
'''
To have a comprehensive comparison,
three classes of popular features are extracted for further
model training: time-domain statistical features (TDF),
frequency-domain statistical features (FDF) and multiple
scale features (MCF).
为了全面比较，提取了三类常用特征进行模型训练:时域统计特征(TDF)、
频域统计特征(FDF)和多尺度特征(MCF)。
'''

import numpy as np
import math
import pandas as pd


# In[1]  Time domin features
def tp1(records):
    """
    平均值,tp1
    """
    return sum(records) / len(records)

def tp2(records):
    """
    均方根值 反映的是有效值而不是平均值,tp2
    """
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))


def tp3(records):
    '''
    绝对值的根植的均值的平方，
    '''
    return (sum([math.sqrt(abs(x)) for x in records])/len(records))**2

def tp4(records):
    '''
    绝对值的均值
    '''
    return sum([abs(x) for x in records]) / len(records)

def tp5(records):
    '''
    3阶中心距
    '''
    return sum([(x-tp1(records))**3 for x in records])/len(records)

def tp6(records):
    '''
    4阶中心距
    '''
    return sum([(x-tp1(records))**4 for x in records])/len(records)

def tp7(records):
    '''
    最大值
    '''
    return max(records)

def tp8(records):
    '''
    最小值
    '''
    return min(records)

def tp9(records):
    '''
    幅值
    '''
    return max(records)-min(records)

def tp10(records):
    '''
    方差的估计（随机变量的方差的估计的分母是n-1）
    '''
    return sum([(x-tp1(records))**2 for x in records])/(len(records)-1)

def tp11(records):
    '''
    均方根除以绝对均值
    '''
    return tp2(records)/tp4(records)

def tp12(records):
    '''
    
    '''
    return tp7(records)/tp2(records)

def tp13(records):
    '''
    
    '''
    return tp7(records)/tp4(records)

def tp14(records):
    '''
    
    '''
    return tp7(records)/tp3(records)

def tp15(records):
    '''
    
    '''
    return tp5(records)/(tp2(records)**3)

def tp16(records):
    '''
    
    '''
    return tp6(records)/(tp2(records)**4)


# In[2]  Frequency domin features

from scipy.fftpack import fft, fftshift, ifft
from scipy.fftpack import fftfreq
import numpy as np
import matplotlib.pyplot as plt

def f(k,records):
    '''
    frequency value,可由fft方法得到
    '''
    Y = fft(records)
    shift_Y = fftshift(Y)
    return None
    
def y(k,recordss):
    '''
    
    '''
    return 
    
def fp1(k,records):
    '''
    
    '''
    return sum(y(records))/k

def fp2(k,records):
    '''
    
    '''
    return  sum((y(k,records)-fp1(k,records))**2)/(k-1)

def fp3(k,records):
    '''
    
    '''
    return sum((y(k,records)-fp1(k,records))**3)/(k*(math.sqrt(fp2(k,records)))**3)

def fp4(k,records):
    '''
    
    '''
    up = sum([(y-fp1)**4 for k in len(records)])
    down = len(records)*(fp2**2)
    return up/down

def fp5(records):
    '''
    
    '''
    up = sum([f(k,records-fp5(records))**2*y(k,records) for k in range(len(records))])
    down = len(records)
    return math.sqrt(up/down)

def fp6(records):
    '''
    
    '''
    up = sum([f(k,records)*y(k,records) for k in range(len(records))])
    down = sum([y(k,records) for k in range(len(records))])
    return up/down

def fp7(records):
    '''
    
    '''
    up = sum([f(k,records)**2*y (k,records)for k in range(len(records))])
    down = sum([y(k,records)for k in range(len(records))])
    return math.sqrt(up/down)
def fp8(records):
    '''
    
    '''
    up = sum([f(k,records)**4*y (k,records)for k in range(len(records))])
    down = sum([f(k,records)**2*y(k,records) for k in range(len(records))])
    return math.sqrt(up/down)

def fp9(records):
    '''
    
    '''
    up = sum([f(k,records)**2*y(k,records)])
    down = math.sqrt(sum([y(k,records) for k in range(len(records))]) * 
                     sum([f(k,records)**4*y(k,records) for k in range(len(records))]))
    return up/down

def fp10(records):
    '''
    
    '''
    return fp6(records)/fp5(records)

def fp11(records):
    '''
    
    '''
    up = sum([(f(k,records)-fp5(k,records))**3*y(k,records) for k in range(len(records))])
    down = len(records)*fp6(records)**3
    return up/down

def fp12(records):
    '''
    
    '''
    up = sum([(f(k,records)-fp5(k,records))**4*y(k,records) for k in range(len(records))])
    down = len(records)*(fp6(records))**4
    return up/down 

def fp13(records):
    '''
    
    '''
    up = sum([math.sqrt(f(k,records)-fp5(records))*y(k,records) for k in range(len(records))])
    down = len(records)*math.sqrt(fp6(records))
    return up/down


# In[3] processing
path_local = r'X:\Work of Lab\单变量时间序列分类_2019_12_20\Step_1_data_preprocess'
df = pd.read_csv(open(path_local + '\\equalized_data.csv'))


#_________________ generate the df_tf_16

df_tf_16 = pd.DataFrame(
        columns = ['tp1','tp2','tp3','tp4','tp5','tp6','tp7','tp8',
                   'tp9','tp10','tp11','tp12','tp13','tp14','tp15','tp16','label'])    
for i in range(df.shape[0]):
    records = np.array(df.iloc[i])
    tp1_ = tp1(records)
    tp2_ = tp2(records)
    tp3_ = tp3(records)
    tp4_ = tp4(records)
    tp5_ = tp5(records)
    tp6_ = tp6(records)
    tp7_ = tp7(records)
    tp8_ = tp8(records)
    tp9_ = tp9(records)
    tp10_ = tp10(records)
    tp11_ = tp11(records)
    tp12_ = tp12(records)
    tp13_ = tp13(records)
    tp14_ = tp14(records)
    tp15_ = tp15(records)
    tp16_ = tp16(records)
    label = records[-1]
    featrue = [tp1_,tp2_,tp3_,tp4_,tp5_,tp6_,tp7_,tp8_,
               tp9_,tp10_,tp11_,tp12_,tp13_,tp14_,tp15_,tp16_,label]
    df_feature = pd.DataFrame(featrue).T
    df_feature.columns = df_tf_16.columns
    df_tf_16 = pd.concat([df_tf_16,df_feature])
#    print(df_feature)
    print(df_tf_16.shape) 
    
df_tf_16.index = df.index
print(df_tf_16.shape)

#df_features.to_csv('features.csv',index=None)

#_________________ generate the df_tf_10
df_tf_10 = df_tf_16.drop(['tp11','tp12','tp13','tp14','tp15','tp16'], axis = 1)

#_________________ generate the df_tf_6
df_tf_6 = df_tf_16.drop(['tp7','tp8','tp9','tp10','tp11','tp12','tp13','tp14','tp15','tp16'], axis = 1)

all_columns = ['tp1','tp2','tp3','tp4','tp5','tp6','tp7','tp8','tp9','tp10','tp11','tp12','tp13','tp14','tp15','tp16','label']  
for i in range(6,17):
    df_tf_i = df_tf_16.drop(all_columns[i:16], axis = 1)
    df_tf_i.to_csv(path_local + '\\data_normalized_tf\\data_normalized_tf_' + str(i)  + '.csv')
    
    
    
# In[4] 逐列归一化并保存文件
df_normalized = df_tf_6
df_normalized = (df_normalized - df_normalized.min()) / (df_normalized.max() - df_normalized.min())
df_normalized['label'] =df_normalized['label'].map(lambda x:int(x*9))

df_normalized.to_csv(path_local + '\\data_normalized_tf_6.csv')

# In[5] devide the sample=
# 分为train和test两文件
train_normalized = df_normalized.sample(frac=0.7)
train_normalized.to_csv(path_local + '\\train_tf_16_normalized.csv',index=None,columns=None)
print(train_normalized.shape)

# get the complementary set of train sample set
index_test = list(df_normalized.index)
for i in list(train_normalized.index):
    index_test.remove(i) 

test_normalized = df_normalized.loc[index_test,:]
test_normalized.to_csv(path_local + '\\test_tf_16_normalized.csv',index=None,columns=None)
print(test_normalized.shape)


