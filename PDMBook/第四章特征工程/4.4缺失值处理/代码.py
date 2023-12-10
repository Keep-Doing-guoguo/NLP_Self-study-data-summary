import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
data = pd.read_csv('缺失值.csv')
imputer = SimpleImputer(strategy='mean')
#使用均值填充缺失值
data['年龄_Imputer'] = imputer.fit_transform(data[['年龄']])
data['工资_Imputer'] = imputer.fit_transform(data[['工资']])
#模型填充
data_predict = data.dropna(subset=['工资'])
age_index = data_predict['年龄'].isna()
#
data_predict_age = data_predict[~age_index]#这个是要进行训练的数据
data_predict_年龄_predict = data_predict[age_index]#这个是要预测的数据

oneHot = OneHotEncoder()
oneHotData = oneHot.fit_transform(data_predict_age[['国家','购买']])
a = oneHotData.toarray()
b = np.array(data_predict_age['工资'])
x_fit = np.hstack((a,b.reshape(-1,1)))
y_fit = data_predict_age['年龄']
#训练线性回归模型
linear = LinearRegression()
linear.fit(x_fit,y_fit)
predict_data = oneHot.transform(data_predict_年龄_predict[['国家','购买']])
c = predict_data.toarray()
d = np.array(data_predict_年龄_predict['工资'])
x_test_fit = np.hstack((c,d.reshape(-1,1)))
y = linear.predict(x_test_fit)


print()