import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = {
    'family':'SimHei',
    'size':20
}

plt.rc('font',**font)#默认配置参数

import pandas as pd

path = '/Users/zhangguowen/Library/Mobile Documents/com~apple~CloudDocs/pythonProject1/PDMBook/第二章回归模型/2.3简单线性回归分析/线性回归.csv'
data = pd.read_csv(path)

#定义自变量，自变量是二维数组。
x = data[['广告费用(万元)']]
#定义因变量，因变量可以是一维，也可以是二维的。
y = data["销售额(万元)"]
#计算相关系数
data['广告费用(万元)'].corr(data['销售额(万元)'])

#广告费用 作为x轴
#销售额   作为y轴 绘制散点图
data.plot('广告费用(万元)', '销售额(万元)', kind='scatter')
plt.show()


#导入sklearn模块中的LinearRegression函数
from sklearn.linear_model import LinearRegression
#使用线性回归模型进行建模
lrmodel = LinearRegression()

lrmodel.fit(x,y)
#查看参数
a = lrmodel.coef_
#查看截距
b = lrmodel.intercept_

data['预测销售额'] = lrmodel.predict(x)
c = data['销售额(万元)'].corr(data['预测销售额'])
d = data['销售额(万元)'].corr(data['预测销售额'])**2

#计算模型的精度
e = lrmodel.score(x,y)

#生成预测所需的自变量数据框
# pX = pd.DataFrame({
#     '广告费用(万元)': [20]
# })
# #对未知的数据进行预测
# lrmodel.predict(pX)

#绘图
plt.figure()
plt.title('advertise and money')
plt.xlabel('the expensive of advertise')
plt.ylabel('xiaoshoue')
plt.scatter(x,y,s=10)
import numpy as np
xs = pd.DataFrame({
    '广告费用(万元)':np.arange(10,30,0.1)
})
ys  = lrmodel.predict(xs)
plt.plot(xs,ys,linewidth=5)
plt.show()
print()