clc
clear all

 
load('chonggou1.mat')
fs=2560;%����Ƶ��
Ts=1/fs;%��������
L=1024;%��������
t=(0:L-1)*Ts;%ʱ������
STA=1; %������ʼλ��
n=1125; %��������
N=1024;%��������
f=(0:N-1)/fs;


for j=0:749
       x=chonggou1(1+300*j:3072+300*j,1);
       X(:,j+1)=abs(hilbert(x));
       f=(0:N-1)/fs;
       f=f';
       time(j+1,:)=time_statistical_compute(X(:,j+1));%%%%%%%��ά�������ʱ����������
       xfft(:,j+1)=fft(X(:,j+1),N)/N;
       y(:,j+1)=2*abs(xfft(:,j+1));
       fre(j+1,:)=fre_statistical_compute(f,y(:,j+1));%%%%%%%��ά�������Ƶ����������
       feature(j+1,:)=[time(j+1,:),fre(j+1,:)];
end
