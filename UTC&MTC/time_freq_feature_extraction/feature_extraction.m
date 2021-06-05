clc
clear all
A=load('E:\������\����\������ȡ\data2.mat');% ��ȡ.DAT�е�����

x=A.normal_data;
x1=A.wear_data;
x2=A.fault_data;

for i=1:375
    x(375+i,:)=x2(i,:);
    x(750+i,:)=x1(i,:);
end



fs=2560;%����Ƶ��
Ts=1/fs;%��������
L=1024;%��������
t=(0:L-1)*Ts;%ʱ������
STA=1; %������ʼλ��
n=1125; %��������
N=1024;%��������

 for j=1:n
       X(:,j)=abs(hilbert(x(j,:)));
       f=(0:N-1)/fs;
       f=f';
       time(j,:)=time_statistical_compute(X(:,j));%%%%%%%��ά�������ʱ����������
       xfft(:,j)=fft(X(:,j),N)/N;
       y(:,j)=2*abs(xfft(:,j));
       fre(j,:)=fre_statistical_compute(f,y(:,j));%%%%%%%��ά�������Ƶ����������
       feature(j,:)=[time(j,:),fre(j,:)];
end

