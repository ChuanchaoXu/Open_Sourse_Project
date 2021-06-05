function [fac]=fre_statistical_compute(f,y)
%%对频域信号进行统计分析
%%2008年10月10日
fre_line_num=max(size(y));
p1 = mean(y);                                                 % 均值频率 特征1,反映频域振动能量的大小；
p2 = sum((y-p1).^2)/fre_line_num;                             % 标准差 特征2，表示频谱的分散或者集中程度；
p3 = sum((y-p1).^3)/(fre_line_num*sqrt(p2^3));                % 特征3，表示频谱的分散或者集中程度；
p4 = sum((y-p1).^4)/(fre_line_num*p2^2);                      % 特征4，表示频谱的分散或者集中程度；
meanf = sum(f.*y)/sum(y);
p5 = meanf;                                                   % 频率中心 特征5，反映主频带位置的变化；
sigma = sqrt(sum((f-meanf).^2.*y)/fre_line_num);
p6 = sigma;                                                   % 特征6，表示频谱的分散或者集中程度；
p7 = sqrt(sum(f.^2.*y)/sum(y));                               % 均方根频率 特征7，反映主频带位置的变化；
p8 = sqrt(sum(f.^4.*y)/sum(f.^2.*y));                          % 特征8，反映主频带位置的变化；
p9 = sum(f.^2.*y)/sqrt(sum(y)*sum(f.^4.*y));                  % 特征9，反映主频带位置的变化；
p10 = sigma/meanf;                                            % 特征10，表示频谱的分散或者集中程度；
p11 = sum((f-meanf).^3.*y)/(sigma.^3*fre_line_num);           % 特征11，表示频谱的分散或者集中程度；
p12 = sum((f-meanf).^4.*y)/(sigma.^4*fre_line_num);           % 特征12，表示频谱的分散或者集中程度；
p13 = sum(sqrt(abs(f-meanf)).*y)/(sqrt(sigma)*fre_line_num);  % 特征13，表示频谱的分散或者集中程度；
fac=[p1;p2;p3;p4;p5;p6;p7;p8;p9;p10;p11;p12;p13];