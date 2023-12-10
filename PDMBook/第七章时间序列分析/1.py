import pandas as pd
data = pd.read_csv('季节性时间序列.csv')

#加载模块
import statsmodels
#执行季节性时间序列分析
tsr = statsmodels.tsa.seasonal_decompose(
    data['总销量'].values,freq=7
)
#绘制季节性时间分解图
#绘制季节性时间分解图
resplot = tsr.plot()
#获取趋势部分
#获取趋势部分
#获取趋势部分
t = tsr.trend
#获取季节性部分
s = tsr.seasonal
#获取随机误差部分
r = tsr.resid

print()