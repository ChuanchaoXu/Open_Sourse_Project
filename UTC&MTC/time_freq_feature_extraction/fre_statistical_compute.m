function [fac]=fre_statistical_compute(f,y)
%%��Ƶ���źŽ���ͳ�Ʒ���
%%2008��10��10��
fre_line_num=max(size(y));
p1 = mean(y);                                                 % ��ֵƵ�� ����1,��ӳƵ���������Ĵ�С��
p2 = sum((y-p1).^2)/fre_line_num;                             % ��׼�� ����2����ʾƵ�׵ķ�ɢ���߼��г̶ȣ�
p3 = sum((y-p1).^3)/(fre_line_num*sqrt(p2^3));                % ����3����ʾƵ�׵ķ�ɢ���߼��г̶ȣ�
p4 = sum((y-p1).^4)/(fre_line_num*p2^2);                      % ����4����ʾƵ�׵ķ�ɢ���߼��г̶ȣ�
meanf = sum(f.*y)/sum(y);
p5 = meanf;                                                   % Ƶ������ ����5����ӳ��Ƶ��λ�õı仯��
sigma = sqrt(sum((f-meanf).^2.*y)/fre_line_num);
p6 = sigma;                                                   % ����6����ʾƵ�׵ķ�ɢ���߼��г̶ȣ�
p7 = sqrt(sum(f.^2.*y)/sum(y));                               % ������Ƶ�� ����7����ӳ��Ƶ��λ�õı仯��
p8 = sqrt(sum(f.^4.*y)/sum(f.^2.*y));                          % ����8����ӳ��Ƶ��λ�õı仯��
p9 = sum(f.^2.*y)/sqrt(sum(y)*sum(f.^4.*y));                  % ����9����ӳ��Ƶ��λ�õı仯��
p10 = sigma/meanf;                                            % ����10����ʾƵ�׵ķ�ɢ���߼��г̶ȣ�
p11 = sum((f-meanf).^3.*y)/(sigma.^3*fre_line_num);           % ����11����ʾƵ�׵ķ�ɢ���߼��г̶ȣ�
p12 = sum((f-meanf).^4.*y)/(sigma.^4*fre_line_num);           % ����12����ʾƵ�׵ķ�ɢ���߼��г̶ȣ�
p13 = sum(sqrt(abs(f-meanf)).*y)/(sqrt(sigma)*fre_line_num);  % ����13����ʾƵ�׵ķ�ɢ���߼��г̶ȣ�
fac=[p1;p2;p3;p4;p5;p6;p7;p8;p9;p10;p11;p12;p13];