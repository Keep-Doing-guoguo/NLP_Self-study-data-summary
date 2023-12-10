#折线图
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

data = pd.read_csv('折线图.csv')
#duir对日期格式进行转换
data['购买日期'] = pd.to_datetime(data['日期'])
mainColor = (42/256,87/256,141/256,1)

plt.figure()
plt.xlabel(
    '购买日期',
    color=mainColor,
)
plt.ylabel(
    '购买用户数',
    color=mainColor,
)
plt.tick_params(axis='x',colors=mainColor)# 设置x轴的刻度线和网格线的属性
plt.tick_params(axis='y',colors=mainColor)# 设置y轴的刻度线和网格线的属性
#顺滑的曲线
plt.plot(data['购买日期'],data['购买用户数'],'-',color=mainColor)
plt.title(
    '购买用户数'
)
plt.show()

#设置子图
fig,ax1 = plt.subplots()
#左边纵轴绘制购买用户数
plt.title('销售情况')
ax1.set_xlabel('购买日期', color=mainColor)
ax1.set_ylabel('购买用户数', color=mainColor)
ax1.tick_params(axis='x', colors=mainColor)
ax1.tick_params(axis='y', colors=mainColor)
p1 = ax1.plot(data['购买日期'],data['购买用户数'],'-',color='blue')

ax2 = ax1.twinx()
ax2.set_xlabel('购买日期', color=mainColor)
ax2.set_ylabel('广告费用', color=mainColor)
ax2.tick_params(axis='x', colors=mainColor)
ax2.tick_params(axis='y', colors=mainColor)
px = ax2.plot(data['购买日期'],data['广告费用'],'-',color='red')


ps = p1 + px
labs = ['购买用户数','广告费用']
ax1.legend(ps,labs)
plt.show()