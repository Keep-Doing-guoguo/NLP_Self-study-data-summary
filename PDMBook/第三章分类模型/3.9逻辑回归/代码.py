import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#读取数据
data = pd.read_csv('逻辑回归.csv')
#查看缺失值信息
a = data.info()
b = data.isnull().sum()
#删除缺失值
data = data.dropna()
#one Hot特征列
features = [
    'Gender', 'Internet Connection', 'Marital Status',
    'Movie Selector', 'Prerec Format', 'TV Signal',
    'Education Level', 'PPV Freq', 'Theater Freq',
    'TV Movie Freq', 'Prerec Buying Freq',
    'Prerec Renting Freq', 'Prerec Viewing Freq'
]
#新建独热编码器
oneHotEncoder = OneHotEncoder()
#训练独热编码器，得到转换规则
oneHotEncoder.fit(data[features])
#转换数据
oneHotData = oneHotEncoder.transform(data[features])

#数值特征列
numericialColumns = [
    'Age', 'Num Bathrooms',
    'Num Bedrooms', 'Num Cars',
    'Num Children', 'Num TVs'
]
cc = type(data[numericialColumns])
cc1 = np.array(data[numericialColumns])
oneHotData = oneHotData.toarray()
x = np.hstack((oneHotData,np.array(data[numericialColumns])))

#
y = data['Home Ownership']

#罗辑回归模型
lrModel = LogisticRegression(max_iter=1000)
cvs = cross_val_score(lrModel,x,y,cv=10)
c = cvs.mean()


print(c)