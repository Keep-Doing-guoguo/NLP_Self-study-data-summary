import torch
import torch as torch
from IPython import display
from d2l import torch as d2l


batch_size = 32
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


#3.6.1 初始化模型参数
num_inputs = 784#28*28=784
num_outputs = 10
#使用正态分布来初始化参数
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

#3.6.2 定义softmax操作
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#保持当前应有的维度
#回顾⼀下sum运算符如何沿着张量中的特定维度⼯作
X.sum(0, keepdim=True), X.sum(1, keepdim=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition # 这⾥应⽤了⼴播机制

#3.6.3 定义模型
def net(X):
    #将线性化之后的结果带入到softmax中再次计算。
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

#3.6.4 定义损失函数
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print(len(y_hat))
print(range(4))
print()
print(range(len(y_hat)))
print(y_hat[(0,1,0,1),(0,1,2,2)])
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

print(cross_entropy(y_hat, y))

#3.6.5 分类精度
def accuracy(y_hat, y): #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
        #print('y_hat: ',y_hat)
    cmp = y_hat.type(y.dtype) == y#y_hat.type(y.dtype)首先进行数据类型的转换。
    #print(cmp.type(y.dtype))
    return float(cmp.type(y.dtype).sum())

a = accuracy(y_hat, y) / len(y)

class Accumulator:  # @save
    """在n个变量上累加"""
    #类的构造函数
    def __init__(self, n):
        self.data = [0.0] * n
    #
    def add(self, *args):
        #循环这里面的a，b然后将其相加
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    #定义的是特殊的方法，内置方法
    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy(net, data_iter): #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval() # 将模型设置为评估模式

    metric = Accumulator(2) # 正确预测数、预测总数

    with torch.no_grad():

        #这里是直接进行来一个预测结果分析
        for X, y in data_iter:
            #返回列表当中一共有多少个元素(1,2,3) 1*2*3
            metric.add(accuracy(net(X), y), y.numel())
            #metric[1]代表的是这一批的数据大小，metric[0]代表的是
            #print('metric[0]: ',metric[0],'metric[1]: ',metric[1])
            '''
            第一次的传入
            0，0  7，32
            0+7 0+32
            第二次传入
            7,32  5,32
            7+5 32+32
            
            
            data：0，0
            '''
    return metric[0] / metric[1]#计算得到的是总损失大小


# print(evaluate_accuracy(net, test_iter))
#每一个epoch的输出
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()#l.mean()计算得到的是每一个图片的损失大小
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
            #32个的损失总和，32个被计算正确的总和。
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()#使用svg格式在Jupyter中显示绘图。
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)#返回的结果，分别是，每一个图片的损失，以及准确度（正确的除以全部的）
        test_acc = evaluate_accuracy(net, test_iter)#没训练一个epoch都就对测试集进行一个验证。
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics#得到的是训练最后一批数据的损失和准确度，理应来说是最小的。
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)