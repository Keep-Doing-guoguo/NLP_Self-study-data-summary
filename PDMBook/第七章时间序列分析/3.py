#时间序列预测
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

data = pd.read_csv('时间序列预测.csv')
#
data.index = pd.to_datetime(data.date,format='%Y%m%d')
#删掉date列，因为已经保存到索引中。
del data['date']
#对数据进行绘图
plt.figure()
plt.plot(data,'blue')
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts

#封装一个方法，方便解读 adfuller 函数的结果
def tagADF(t):
    result = pd.DataFrame(index=[
            "Test Statistic Value",
            "p-value", "Lags Used",
            "Number of Observations Used",
            "Critical Value(1%)",
            "Critical Value(5%)",
            "Critical Value(10%)"
        ], columns=['value']
    )
    result['value']['Test Statistic Value'] = t[0]
    result['value']['p-value'] = t[1]
    result['value']['Lags Used'] = t[2]
    result['value']['Number of Observations Used'] = t[3]
    result['value']['Critical Value(1%)'] = t[4]['1%']
    result['value']['Critical Value(5%)'] = t[4]['5%']
    result['value']['Critical Value(10%)'] = t[4]['10%']
    return result
#使用ADF单位根检验法，检验时间序列的稳定性
adf_Data = ts.adfuller(data.value)
#解读 ADF 单位根检验法得到的结果
adfResult = tagADF(adf_Data)


print()

