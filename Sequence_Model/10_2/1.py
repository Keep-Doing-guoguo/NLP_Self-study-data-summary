import torch
from torch import nn
from d2l import torch as d2l


n_train = 50  # 训练样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5)   # 排序后的训练样本,然后对其放大5倍。

def f(x):
    return 2 * torch.sin(x) + x**0.8#数学公式：2*sin + x的0.8次方

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出，后面的torch.normal(0.0, 0.5, (n_train,))应该为噪声
x_test = torch.arange(0, 5, 0.1)  # 测试样本
y_truth = f(x_test)  # 测试样本的真实输出
n_test = len(x_test)  # 测试样本数
n_test


def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5);
    d2l.plt.show()


y_hat = torch.repeat_interleave(y_train.mean(), n_test)#重复张量的元素
# plot_kernel_reg(y_hat)
print()



X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))# X_repeat的形状:(n_test,n_train),# 每一行都包含着相同的测试输入（例如：同样的查询）


a = -(X_repeat - x_train)**2 / 2
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)# 每一行都包含着要在给定的每个查询的值（y_train）之间分配的注意力权重# x_train包含着键。attention_weights的形状：(n_test,n_train),

y_hat = torch.matmul(attention_weights, y_train)# y_hat的每个元素都是值的加权平均值，其中的权重是注意力权重
# plot_kernel_reg(y_hat)#再进行画图显示。在这里只是乘上了一个注意力权重就变得更加接近真实值了。
a1 = attention_weights.unsqueeze(0).unsqueeze(0)#增加维度信息
# d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),xlabel='Sorted training inputs',ylabel='Sorted testing inputs')#显示热力图


X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))
a2 = torch.bmm(X, Y).shape#批量矩阵乘法a

weights = torch.ones((2, 10)) * 0.1
values = torch.arange(20.0).reshape((2, 10))
b = weights.unsqueeze(1)
b1 = values.unsqueeze(-1)
a3 = torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))

class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))#就单单一个值

    def forward(self, queries, keys, values):
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # values的形状为(查询个数，“键－值”对个数)
        a = self.attention_weights.unsqueeze(1)
        a1 = values.unsqueeze(-1)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)#这是和y进行相乘，然后转换维度。是和书中的公式相同的。



X_tile = x_train.repeat((n_train, 1))# X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入

Y_tile = y_train.repeat((n_train, 1))# Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
b2= torch.eye(n_train)#对角线全部是1，维度和train相同
b3 = (1 - torch.eye(n_train)).type(torch.bool)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))# keys的形状:('n_train'，'n_train'-1),除了对角线上的元素不取，取对角线之外的元素。

values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))# values的形状:('n_train'，'n_train'-1)。和上面相似。所以总共有2450个元素。


net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    a = net(x_train, keys, values)
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))



keys = x_train.repeat((n_test, 1))# keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）

values = y_train.repeat((n_test, 1))# value的形状:(n_test，n_train)
y_hat = net(x_test, keys, values).unsqueeze(1).detach()#k和v相当于权重参数
plot_kernel_reg(y_hat)

d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')





