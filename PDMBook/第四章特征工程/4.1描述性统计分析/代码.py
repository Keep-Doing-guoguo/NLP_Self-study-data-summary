import pandas as pd
data = pd.read_csv('描述性统计分析.csv')

a = data['注册时长'].describe()
b = data['注册时长'].hist()#画出来直方图

print()