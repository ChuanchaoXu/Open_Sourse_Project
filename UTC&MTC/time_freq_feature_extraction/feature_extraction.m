clc
clear all
A=load('E:\大论文\程序\特征提取\data2.mat');% 读取.DAT中的数据

x=A.normal_data;
x1=A.wear_data;
x2=A.fault_data;

for i=1:375
    x(375+i,:)=x2(i,:);
    x(750+i,:)=x1(i,:);
end



fs=2560;%采样频率
Ts=1/fs;%采样周期
L=1024;%采样点数
t=(0:L-1)*Ts;%时间序列
STA=1; %采样起始位置
n=1125; %样本个数
N=1024;%样本点数

 for j=1:n
       X(:,j)=abs(hilbert(x(j,:)));
       f=(0:N-1)/fs;
       f=f';
       time(j,:)=time_statistical_compute(X(:,j));%%%%%%%多维矩阵计算时域特征矩阵
       xfft(:,j)=fft(X(:,j),N)/N;
       y(:,j)=2*abs(xfft(:,j));
       fre(j,:)=fre_statistical_compute(f,y(:,j));%%%%%%%多维矩阵计算频域特征矩阵
       feature(j,:)=[time(j,:),fre(j,:)];
end

