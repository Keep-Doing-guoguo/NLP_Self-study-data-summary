
#3.3.1. 生成数据集
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

#3.3.2. 读取数据集
#data_arrays既包括X也包括Y
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    #主要是对数据进行batch的划分
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
print(next(iter(data_iter)))


#3.3.3. 定义模型
# nn是神经网络的缩写
from torch import nn
#定义的是线性模型
net = nn.Sequential(nn.Linear(2, 1))

#3.3.4. 初始化模型参数
#对第1层的参数进行初始化。normal_标准正态分布，方差是0，标准差是0.01
net[0].weight.data.normal_(0, 0.01)
#设置偏差是0
net[0].bias.data.fill_(0)

#3.3.5. 定义损失函数
#定义均方误差损失函数
loss = nn.MSELoss()

#3.3.6. 定义优化算法
#小批量随机梯度下降算法是一种优化神经网络的标准工具，PyTorch在optim模块中实现了该算法的许多变种
#parameters所定义的是优化的参数
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

#3.3.7. 训练
num_epochs = 3
for epoch in range(num_epochs):
    running_loss = 0.0
    for X, y in data_iter:
        #net(X)可以计算出来Y_pre，
        l = loss(net(X) ,y)
        #将梯度归0
        trainer.zero_grad()
        #进行反向梯度
        l.backward()
        #来进行模型更新的。
        trainer.step()
        running_loss += l.item()

    #chong重新再次计算一次loss
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

#误差
w = net[0].weight.data#得到的应该是一个1行2列的向量
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data#得到的一个标量
print('b的估计误差：', true_b - b)