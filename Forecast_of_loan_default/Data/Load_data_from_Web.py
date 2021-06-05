# 从tmp网盘加载文件进行查看数据（限本地jupyter notebook）
import csv
import pandas as pd
from io import StringIO
from urllib import request
import pandas as pd

# 基于tmp.link分享链接直接加载文件
def get_data_from_tmp_web(url):
  file_mark = url[-13:]
  base_url = 'https://tmplinkapp-connect.vx-cdn.com/connect-tz6rhexhflovcrsiq7w5-'
  download_link = base_url + file_mark

  s = request.urlopen(download_link).read().decode('utf8')  # 1 读取数据串

  dfile = StringIO(s)      # 2 将字符串转换为 StringIO对象，使其具有文件属性 
  creader = csv.reader(dfile)  # 3 将流 转换为可迭代的 reader（csv row）
  dlists=[rw for rw in creader]  # 4 其他转换、操作

  temp = []
  for i in range(len(dlists)):
    row_int = [float(j) for j in dlists[i][0].split( )]
    temp.append(row_int)

  df = pd.DataFrame(temp, columns=['voltage_input', 'voltage_output', 'value_tachometer'])
  # print(df.head())

  return df

Train_data = get_data_from_tmp_web('http://tmp.link/f/60b08b073159b')
Test_data = get_data_from_tmp_web('http://tmp.link/f/60b08a85839df')

# 也可以直接基于链接下载到本地
'''
train.csv  -->  http://tmp.link/f/60b08b073159b
testA.csv  -->  http://tmp.link/f/60b08a85839df
sample_submit.csv  --> http://tmp.link/f/60b08a493aeb0
'''


