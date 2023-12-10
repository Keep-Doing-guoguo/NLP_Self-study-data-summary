# -*- coding: utf-8 -*-

import pandas
import matplotlib.pyplot as plt

data = pandas.read_csv(
    'DBSCAN.csv',
    encoding='utf8', engine='python'
)

plt.figure()
#绘制散点图
plt.scatter(
    data['x'],
    data['y']
)


from sklearn.cluster import DBSCAN

#设置DBSCAN聚类参数
eps = 0.5
min_samples = 5

model = DBSCAN(eps = eps, min_samples = min_samples)

data['type'] = model.fit_predict(
    data[['x', 'y']]
)

plt.figure()
#画出非噪声点的数据，颜色由聚类分组决定
plt.scatter(
    data[data.type!=-1]['x'],
    data[data.type!=-1]['y'],
    c=data[data.type!=-1]['type']
)
#画出噪声点的数据，用红色的x表示噪声点
plt.scatter(
    data[data.type==-1]['x'],
    data[data.type==-1]['y'],
    c='red', marker='x'
)
plt.show()