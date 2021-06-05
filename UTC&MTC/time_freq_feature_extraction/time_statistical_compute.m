function valfactor=time_statistical_compute(x)

%%2008年10月10日
%%对时域信号进行统计量分析
%% val返回有量纲指标，factor返回无量纲指标
N=length(x);
p1=mean(x); %均值
x=x-p1;
p2=sqrt(sum(x.^2)/N); %均方根值
p3=(sum(sqrt(abs(x)))/N).^2; %方根幅值
p4=sum(abs(x))/N; %绝对平均值
p5=sum(x.^3)/N; %偏斜度
% p6=sum(x.^4)/N; %峭度
p6=kurtosis(x); %峭度
% p7=sum((x).^2)/N; %方差
p7=var(x); %方差
p8=max(x);%最大值
p9=min(x);%最小值
p10=p8-p9;%峰峰值
val=[p1; p2; p3; p4; p5; p6; p7; p8; p9; p10];
%%以上都是有量纲统计量，以下是无量纲统计量
f1=p2/p4; %波形指标
f2=p8/p2; %峰值指标 E[MAX(X)]=P8?
f3=p8/p4; %脉冲指标
f4=p8/p3; %裕度指标
f5=p5/((sqrt(p7))^3); %偏斜度指标
f6=p6/((sqrt(p7))^4); %峭度指标
factor=[f1; f2; f3; f4; f5; f6];
valfactor=[val;factor];