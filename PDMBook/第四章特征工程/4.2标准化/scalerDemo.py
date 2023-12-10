import pandas as pd
from sklearn.preprocessing import scale

data = pd.read_csv('华南地区.csv')

#特征变量
x = data[['注册时长','营收收入','成本']]
#目标变量
y = data['是否续约']

#均值为0，标准差为1进行标准化处理。正态分布。
scalerX = scale(x)


print()