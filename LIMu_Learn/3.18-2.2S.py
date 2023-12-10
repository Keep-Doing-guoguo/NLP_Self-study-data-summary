import os

##1.读取数据集
#首先我们创建一个人工数据集
os.makedirs(os.path.join('../..', 'data'), exist_ok=True)
data_file = os.path.join('../..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    # write写入字符串到文件描述符 fd中. 返回实际写入的字符串长度
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

#从创建的CSV文件中加载原始数据集，我们导入pandas包并调用read_csv函数
#这其实是一个3行 4列 的数据
import pandas as pd
data=pd.read_csv(data_file)
print(data)

##2.处理缺失值
#插值法（替代插入） 和 缺失值（忽略缺失值）
#iloc=index location 第0和第一列放在input里面 第1列放在output里面
#               前面代表的是行 后面代表的是列      前面代表的是行 后面代表的是列
#               所有的行 和 第0列和第1列         所有的行 和 第1列
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
#对所有是nan的数据填写一个mean 也就是平均值
#inputs = inputs.fillna(inputs.mean()) 在这里报警告的原因是因为后面的东西并不为数字
inputs = inputs.fillna(inputs.mean())
print(inputs)

# pandas可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

##3.转换为张量格式
#现在inputs和outputs中的所有条目都是数值类型
import torch

#X=torch.tensor(inputs.values) y=torch.tensor(outputs.values)
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X)
print(y)