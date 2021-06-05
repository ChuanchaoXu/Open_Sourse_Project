# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 11:26:46 2019

kNN

@author: DELL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
from pyts.approximation import PiecewiseAggregateApproximation
from pyts.approximation import SymbolicAggregateApproximation
from pyts.approximation import DiscreteFourierTransform
from pyts.approximation import MultipleCoefficientBinning
from pyts.approximation import SymbolicFourierApproximation
import time


time_start=time.time()

path = '/content/drive/My Drive/STSC_2019_12_20/Step_2_single-TSC0/Step_1_data_preprocess/equalized_data.csv'



# In[0] 设置path和superparameters
# set path 
path_data = '/content/drive/My Drive/STSC_2019_12_20/Step_2_single-TSC/Step_1_data_preprocess/equalized_data.csv'
df_high_D = pd.read_csv(open(path_data), index_col = None) 
df_high_D_series = df_high_D.iloc[:,:-1]
df_high_D_label = df_high_D.iloc[:,-1]


# In[1] Series dimensionality reduction 
# Paa transformation
window_size = 1
high_D = 6000
low_D = high_D//window_size

# PAA transformation
paa = PiecewiseAggregateApproximation(window_size=window_size)
array_low_D = paa.fit_transform(df_high_D.values)  
df_low_D_series = pd.DataFrame(array_low_D, columns = range(low_D+1),index = df_high_D.index)
#print(df_1000_series)
df_low_D_label = df_high_D_label
df_low_D = pd.concat([df_low_D_series,df_low_D_label], axis = 1)
#print(df_1000) 


# In[2]  image generation by GAF, image show, remove the white margin, save
X = df_low_D
image_size = 1024
image_size_ = int(128/8)
gadf = GramianAngularField(image_size=image_size, method='difference')
X_gadf = gadf.fit_transform(X)
print(X_gadf.shape)

X_transform = X_gadf

path_local = '/content/drive/My Drive/STSC_2019_12_20/Step_2_single-TSC/Step_2_single-TSC/Images_Remarks'
path_train_remark = path_local + '/dataset_train_remark.txt'
dataset_train_remark = open(path_train_remark,'w')

path_test_remark = path_local + '/dataset_test_remark.txt'
dataset_test_remark = open(path_test_remark,'w')

path_train_image =  path_local + '/Images_train/'
path_test_image = path_local + '/Images_test/'

# set the image style
for i in range(df_low_D.shape[0]):  
    number = '%05d' % i # 将每行数据另存为 .csv文件（对每个变量的时间序列另存，去掉header和columns）        
    label = X.iloc[i].tolist()[-1] 
    label = '%d' % label 
    if i < int(0.7*df_low_D.shape[0]): 
        # build the dataset_remark.txt
        remark = path_train_image + number +'.jpg'+ '  ' + label # remark包括图片地址和标签
        dataset_train_remark.write(remark + '\n')
        
        # output and save figures 
#        plt.figure(figsize=(2, 2))
        plt.figure(i)
        plt.imshow(X_transform[i],cmap ='rainbow',origin='lower')
        plt.axis('off')
        
        plt.gcf().set_size_inches(image_size/100.0/3.0, image_size/100.0/3.0)  # dpi=300的情况下，像素和英寸的换算是每英寸≈0.003333像素
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.margins(0,0)
        plt.savefig(path_train_image+number+'.jpg',dpi=300,transparent=True, pad_inches = 0)
        
    else:
        remark = path_test_image + number +'.jpg'+ '  ' + label # remark包括图片地址和标签
        dataset_test_remark.write(remark + '\n')
        
        # output and save figures 
        plt.figure(figsize=(2, 2)) 
        plt.figure(i)
        plt.imshow(X_transform[i],cmap='rainbow',origin='lower')
        plt.axis('off')
        
        plt.gcf().set_size_inches(image_size/100.0/3.0, image_size/100.0/3.0)  # dpi=300的情况下，像素和英寸的换算是每英寸≈0.003333像素
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.margins(0,0)
        
        plt.savefig(path_test_image+number+'.jpg',dpi=300,transparent=True, pad_inches = 0)
    
#    plt.close('all')
       
dataset_train_remark.close()
dataset_test_remark.close()
time_end=time.time()
print('totally cost',time_end-time_start)

# In[3]  CNN Classifier

path_CNN = '/content/drive/My Drive/STSC_2019_12_20/Step_2_single-TSC/Step_2_single-TSC/Images_Remarks'
#import image_classification_cuda 

import torch
from torch.autograd import Variable
import torch.nn as nn
#import torchvision
from torch.utils.data import Dataset, DataLoader  
from torchvision import transforms, utils
from PIL import Image
 
 
# 判定GPU是否存在
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# 定义超参数

num_epochs = 2
batch_size = 1
learning_rate = 0.001
 
def default_loader(path):  
    # 注意要保证每个batch的tensor大小时候一样的。  
    return Image.open(path).convert('RGB')  

'''
class MyDataset(Dataset):  
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):  
        fh = open(txt, 'r')
        imgs = []
        for line in fh:  
            line = line.strip('\n')
            line = line.rstrip()
            line = line.strip()
            words = line.split('  ')
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs   
        self.transform = transform   
        self.target_transform = target_transform  
        self.loader = loader
      
    def __getitem__(self, index):
        fn, label = self.imgs[index]  
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label
      
    def __len__(self):  
        return len(self.imgs)  
'''

class MyDataset(Dataset):  
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):  
        fh = open(txt, 'r')  
        imgs = []
        for line in fh:  
            line = line.strip()  
            line = line.strip('\n')
            # line = line.rstrip()  
            words = line.split()  
            words = line.split('  ')
            words_ = (words[0], int(words[1]))
            imgs.append(words_)  
        self.imgs = imgs  
        self.transform = transform  
        self.target_transform = target_transform  
        self.loader = loader  
      
    def __getitem__(self, index):
        fn, label = self.imgs[index]  
        img = self.loader(fn)  
        if self.transform is not None:  
            img = self.transform(img)  
        return img,label  
    
    def __len__(self):  
        return len(self.imgs) 



root = path_CNN
train_data = MyDataset(txt = root+'/dataset_train_remark.txt', transform = transforms.ToTensor())
test_data = MyDataset(txt = root+'/dataset_test_remark.txt', transform = transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size = batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size = batch_size) 



# build CNN
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.Dropout (p=0.5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.Dropout (p=0.5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, 1, 1),
            torch.nn.Dropout (p=0.5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, 1, 1),
            torch.nn.Dropout (p=0.5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, 3, 1, 1),
            torch.nn.Dropout (p=0.5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, 3, 1, 1),
            torch.nn.Dropout (p=0.5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(1024* 16 *16, 512), #如果需要修改图片尺寸，则需要修改此处的输入数据的维度
            torch.nn.Dropout (p=0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10) 
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        res = conv4_out.view(conv4_out.size(0), -1)
        out = self.dense(res)
        return out
            

model = Net().to(device)
#print(model)

optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()



Acc_train = []
Loss_train = []

Acc_test = []
Loss_test = []

for epoch in range(num_epochs):
    epoch = 0
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = Variable(batch_x).to(device), Variable(batch_y).to(device)
        out = model(batch_x)
#        print(out.size)
        loss = loss_func(out, batch_y)
#        train_loss += loss.data[0]
        train_loss += loss.item()  
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
#        train_acc += train_correct.data[0]
        train_acc += train_correct.item()  
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
    Loss_train.append(train_loss/len(train_data))
    Acc_train.append(100*train_acc/len(train_data))
    
    print('Train Loss: {:.6f}, Acc: {:.6f}%'.format(train_loss/len(train_data),
          100*train_acc/len(train_data)))

    # evaluation--------------------------------
    model.eval()
    eval_loss = 0.
    eval_acc = 0. 
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = Variable(batch_x, volatile=True).to(device), Variable(batch_y, volatile=True).to(device)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
#        eval_loss += loss.data[0]
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
#        eval_acc += num_correct.data[0]
        eval_acc += num_correct.item()
        
        
    Loss_test.append(eval_loss/len(test_data))
    Acc_test.append(100*eval_acc/len(test_data))
    
    
    print('Test Loss: {:.6f}, Acc: {:.6f}%'.format(eval_loss/len(test_data),
          100*eval_acc/len(test_data)))
#    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
#            test_data)), eval_acc / (len(test_data))))
    
    

# 保存模型参数
torch.save(model.state_dict(), 'model.ckpt')


#import matplotlib.pyplt as plt


















