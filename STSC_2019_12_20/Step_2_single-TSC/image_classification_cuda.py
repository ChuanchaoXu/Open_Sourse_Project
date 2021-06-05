# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 19:11:32 2019

@author: XI
"""

# coding=utf-8
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader  
from torchvision import transforms, utils  
from PIL import Image
 
 
# 判定GPU是否存在
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# 定义超参数

num_epochs = 50
batch_size = 10
learning_rate = 0.001
 
def default_loader(path):  
    # 注意要保证每个batch的tensor大小时候一样的。  
    return Image.open(path).convert('RGB')  
  
class MyDataset(Dataset):  
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):  
        fh = open(txt, 'r')  
        imgs = []  
        for line in fh:  
            line = line.strip()  
            line = line.strip('\n')
            # line = line.rstrip()  
            words = line.split()  
            
            line = line.strip()  
            line = line.strip('\n')
            # line = line.rstrip()  
            words = line.split()
            
            imgs.append((words[0],int(words[1])))  
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



root = r'E:\寒假\代码相关\单变量时间序列分类_2019_12_20\Step_2_single-TSC\Images_Remarks'
train_data=MyDataset(txt=root+'\dataset_train_remark.txt', transform=transforms.ToTensor())
test_data=MyDataset(txt=root+'dataset_test_remark.txt', transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size)



# build CNN
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 5, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 5, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64* 25 *25, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10) 
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out


model = Net().to(device)
#print(model)

optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
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
        
    print('Train Loss: {:.6f}, Acc: {:.6f}%'.format(eval_loss/len(test_data),
          100*eval_acc/len(test_data)))
#    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
#            test_data)), eval_acc / (len(test_data))))
    
# 保存模型参数
torch.save(model.state_dict(), 'model.ckpt')





















































