#%matplotlib inline
import random
import torch
from d2l import torch as d2l
#1生成数据集
'''
我们生成一个包含1000个样本的数据集
合成数据集是一个矩阵1000行2列的矩阵

'''
#num_example传入的num_example指的就是数字的长度
def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    #均差是0，标准差是1 输出到num_examples 中 ，长度是len（w）
    #(0,1)指的是正态分布的随机数
    #(num_examples, len(w))指的是该X的矩阵的行有1000行，列有2列
    X = torch.normal(0, 1, (num_examples, len(w)))

    #print(X) 1000*2
    #print(len(X))  1000
    #torch.matmul是tensor的乘法，输入可以是高维的。

    #输出的是一个1000行1列的矩阵
    y = torch.matmul(X, w) + b

    #print(y)
    #print(len(y))
    #print(y[0])


    #均差是0，标准差是0.01 输出的大小是y的形状
    y += torch.normal(0, 0.01, y.shape)

    #将y的大小进行修改。
    return X, y.reshape((-1, 1))


#这个是w向量
true_w = torch.tensor([2, -3.4])

print(true_w)

#这个是偏差
true_b = 4.2

#到这里才变成了一个1000*1
#1000*2 1000*1
features, labels = synthetic_data(true_w, true_b, 1000)

#print(features[0])
#print(len(labels[0]))
#print(len(labels))

#画图
d2l.set_figsize()
#scatter里面的两个参数 后面的1是设置斑点的大小的
#features[:, (1)]代表的是所有的行 第一列的数据与labels进行相互对应
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1);
#d2l.plt.show()

#print(features[:, (1)].detach().numpy())
#print(labels.detach().numpy())
print(list(range(1000)))

#读取数据集
#自定义函数
#定义大小 ，1000*2 1000*1
def data_iter(batch_size, features, labels):

    #行数是1000
    num_examples = len(features)

    #创建一个从0-999的列表，对应数据的索引
    indices = list(range(num_examples))

    # 这些样本是随机读取的，没有特定的顺序
    #将列表进行了顺序的打乱
    random.shuffle(indices)

    #随机进行读取数据
    #range(start, stop[, step])
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break