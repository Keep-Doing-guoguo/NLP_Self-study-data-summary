import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

path = '/Users/zhangguowen/Library/Mobile Documents/com~apple~CloudDocs/pythonProject1/PDMBook/第二章回归模型/2.5一元非线性回归/一元非线性回归.csv'
data = pd.read_csv(path)

x = data[['发布天数']]
y = data[["活跃用户数"]]

#新建一个绘图窗口
plt.figure()
#设置图形标题
plt.title('days_peoples')
#设置x轴标签
plt.xlabel('tian_shu')
#设置y轴标签
plt.ylabel('huo_yue_yong_hu_shu')
plt.scatter(x,y)
plt.show()#从这里能够看出来，这个数据是非线性数据。

#尝试二元线性回归，创建数据特征。
data['x0'] = data['发布天数'].pow(0)
data['x1'] = data['发布天数'].pow(1)
data['x2'] = data['发布天数'].pow(2)

x = data[['x0','x1','x2']]
y = data[['活跃用户数']]

lrModel = LinearRegression()
lrModel.fit(x,y)
a = lrModel.score(x,y)

data['x3'] = data['发布天数'].pow(3)

x = data[['x0','x1','x2','x3']]
y = data[['活跃用户数']]

lrModel = LinearRegression()
lrModel.fit(x,y)
b = lrModel.score(x,y)

polyno = PolynomialFeatures(degree=4)
x = data[['发布天数']]
x_4 = polyno.fit_transform(x)

x = data[['发布天数']]
ds = []
scores = []
for d in range(2,20):
    ds.append(d)

    polyno = PolynomialFeatures(degree=d)
    x_d = polyno.fit_transform(x)#在这里是获取到增加的数据信息

    lrModel = LinearRegression()
    lrModel.fit(x_d,y)

    scores.append(lrModel.score(x_d,y))

dScores = pd.DataFrame({
    'degree':ds,
    'socres':scores
})

#新建一个绘图窗口
plt.figure()
#是这图形标题
plt.title(
    'scores'
)
#
plt.xlabel('degree')
plt.ylabel('score')
plt.scatter(ds,scores)
plt.show()

x = data[['发布天数']]
y = data[['活跃用户数']]
polyno = PolynomialFeatures(degree=3)
x_3 = polyno.fit_transform(x)
lrModel = LinearRegression()
lrModel.fit(x_3,y)
socre = lrModel.score(x_3,y)

px = pd.DataFrame({
    '发布天数':[730]
})
px_3 = polyno.transform(px)
f = lrModel.predict(px_3)


print()
