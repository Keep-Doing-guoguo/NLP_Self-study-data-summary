import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

data = pd.read_csv('华南地区.csv')
power = PowerTransformer()
#特征变量
x = data[['注册时长', '营收收入', '成本']]
#目标变量
y = data['是否续约']
px = power.fit_transform(x)

gauss = GaussianNB()
cvs1 = cross_val_score(gauss,x,y,cv=3).mean()


gauss = GaussianNB()
cvs2 = cross_val_score(gauss,px,y,cv=3).mean()


import numpy as np
# 构造一组数据
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 实例化PowerTransformer对象
pt = PowerTransformer()

# 拟合数据
pt.fit(X)

# 进行幂次变换
X_transformed = pt.transform(X)
print()