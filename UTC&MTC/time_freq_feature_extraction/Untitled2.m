clc
clear all

 
load('chonggou1.mat')
fs=2560;%采样频率
Ts=1/fs;%采样周期
L=1024;%采样点数
t=(0:L-1)*Ts;%时间序列
STA=1; %采样起始位置
n=1125; %样本个数
N=1024;%样本点数
f=(0:N-1)/fs;


for j=0:749
       x=chonggou1(1+300*j:3072+300*j,1);
       X(:,j+1)=abs(hilbert(x));
       f=(0:N-1)/fs;
       f=f';
       time(j+1,:)=time_statistical_compute(X(:,j+1));%%%%%%%多维矩阵计算时域特征矩阵
       xfft(:,j+1)=fft(X(:,j+1),N)/N;
       y(:,j+1)=2*abs(xfft(:,j+1));
       fre(j+1,:)=fre_statistical_compute(f,y(:,j+1));%%%%%%%多维矩阵计算频域特征矩阵
       feature(j+1,:)=[time(j+1,:),fre(j+1,:)];
end
