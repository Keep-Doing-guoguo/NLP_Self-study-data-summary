import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

font = {
    'family':'SimHei',
    'size':20
}
plt.rc('font',**font)
path = '/Users/zhangguowen/Library/Mobile Documents/com~apple~CloudDocs/pythonProject1/PDMBook/第二章回归模型/2.4多重线性回归分析/线性回归.csv'
data = pd.read_csv(path)
#定义自变量
x = data[["广告费用(万元)", "客流量(万人次)"]]
#定义因变量
y = data[['销售额(万元)']]

#计算相关系数
a = data['广告费用(万元)'].corr(data['销售额(万元)'])
b = data['客流量(万人次)'].corr(data['销售额(万元)'])

#广告费用 作为x轴
#销售额 作为y轴 绘制散点图
fig ,axs = plt.subplots(1,2)
axs[0].scatter(data['广告费用(万元)'],data['销售额(万元)'])
axs[1].scatter(data['客流量(万人次)'],data['销售额(万元)'])
plt.show()

#使用线性回归模型进行建模
lrModel = LinearRegression()
lrModel.fit(x,y)
#查看参数
c = lrModel.coef_
#查看截距
d = lrModel.intercept_
#查看准确度
e = lrModel.score(x,y)
#生成预测所需要的自变量数据框
pX = pd.DataFrame({
    '广告费用(万元)': [20,21],
     '客流量(万人次)': [5,6]
})
f = lrModel.predict(pX)
print()