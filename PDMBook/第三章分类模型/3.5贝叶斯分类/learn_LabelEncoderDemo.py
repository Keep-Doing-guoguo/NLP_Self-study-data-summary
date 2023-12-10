import pandas as pd
from sklearn.preprocessing import OneHotEncoder
#读取案例数据到data变量中
data = pd.read_csv('LabelEncoder.csv')
#onehot = OneHotEncoder(drop='first')
onehot = OneHotEncoder()
data_one = onehot.fit_transform(data)
a = data_one.toarray()
'''
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [1. 1. 0.]
 [0. 0. 0.]]
'''
print(a)