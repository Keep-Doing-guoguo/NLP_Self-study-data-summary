import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

#4.4.4.1. 生成数据集
max_degree = 20  # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5,1.2,-3.4,5.6])#也就是说明了这是一个3次多项式


#生成标准分布 200行1列 200个数据1个特征
features = np.random.normal(size=(n_test+n_train,1))

#对其进行打乱顺序
np.random.shuffle(features)
#np.power()用于数组元素求n次方。将每一个features进行（0，1，2---19）进行次方
poly_features=np.power(features,np.arange(max_degree).reshape(1,-1))
#生成一个(200, 20)的数据

#生成1行20列的数据

for i in range(max_degree):#对每一个数字进行了这样子的操作
    #[:,i]代表的是所有行，与0列，1列，2列----19列
    poly_features[:,i]/=math.gamma(i+1)# gamma(n)=(n-1)!


# labels的维度:(n_train+n_test,)
labels = np.dot(poly_features, true_w)
#labels += np.random.normal(scale=0.1, size=labels.shape)#这个是生成的噪音标签数据



# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]

features[:2], poly_features[:2, :], labels[:2]
def evaluate_loss(net, data_iter, loss):  #@save
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = nn.MSELoss(reduction='none')# none 不求平均 # 默认为mean #sum
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    # animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
    #                         xlim=[1, num_epochs], ylim=[1e-3, 1e2],
    #                         legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        # if epoch == 0 or (epoch + 1) % 20 == 0:
        #     animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
        #                              evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())


# 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])
#[5,1.2,-3.4,5.6
#weight: [[ 4.9870768  1.1666485 -3.369597   5.6550126]]#
#weight: [[ 4.9999275  1.208778  -3.4007885  5.581698 ]]#去掉噪声之后。