import pandas as pd

path = '/Users/zhangguowen/Library/Mobile Documents/com~apple~CloudDocs/pythonProject1/PDMBook/第二章回归模型/2.2相关分析/相关分析.csv'
data = pd.read_csv(path,engine='python',encoding='utf-8')

relation = data['人口'].corr(data['文盲率'])#查看相关系数，corr是pandas中自带函数。

corrMatrix = data[['超市购物率','网上购物率','文盲率','人口']].corr()

print()