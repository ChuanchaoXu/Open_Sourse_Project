function valfactor=time_statistical_compute(x)

%%2008��10��10��
%%��ʱ���źŽ���ͳ��������
%% val����������ָ�꣬factor����������ָ��
N=length(x);
p1=mean(x); %��ֵ
x=x-p1;
p2=sqrt(sum(x.^2)/N); %������ֵ
p3=(sum(sqrt(abs(x)))/N).^2; %������ֵ
p4=sum(abs(x))/N; %����ƽ��ֵ
p5=sum(x.^3)/N; %ƫб��
% p6=sum(x.^4)/N; %�Ͷ�
p6=kurtosis(x); %�Ͷ�
% p7=sum((x).^2)/N; %����
p7=var(x); %����
p8=max(x);%���ֵ
p9=min(x);%��Сֵ
p10=p8-p9;%���ֵ
val=[p1; p2; p3; p4; p5; p6; p7; p8; p9; p10];
%%���϶���������ͳ������������������ͳ����
f1=p2/p4; %����ָ��
f2=p8/p2; %��ֵָ�� E[MAX(X)]=P8?
f3=p8/p4; %����ָ��
f4=p8/p3; %ԣ��ָ��
f5=p5/((sqrt(p7))^3); %ƫб��ָ��
f6=p6/((sqrt(p7))^4); %�Ͷ�ָ��
factor=[f1; f2; f3; f4; f5; f6];
valfactor=[val;factor];