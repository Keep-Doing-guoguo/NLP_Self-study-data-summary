import torch
from torch import nn
from d2l import torch as d2l

'''
in_channels —— 输入的channels数

out_channels —— 输出的channels数

kernel_size ——卷积核的尺寸，可以是方形卷积核、也可以不是，下边example可以看到

stride —— 步长，用来控制卷积核移动间隔

padding ——输入边沿扩边操作

且所有的操作都是基于（W-F+2P）/S+1计算出来的下一个层的大小，以及需要填充的padding
'''

#所定义出来的LeNet-5
net = nn.Sequential(
    #28*28变为28*28 通道数1-6
    nn.Conv2d(1,6,kernel_size=5,padding=2),nn.Sigmoid(),
    #经过池化层缩小2倍。
    nn.AvgPool2d(kernel_size=2,stride=2),

    #增加通道数，变成16个10*10
    nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),

    #池化层之后变成16个5*5
    nn.AvgPool2d(kernel_size=2,stride=2),
    #两个卷积层，卷积之后必须要加上sigmoid函数。接下来就是3个全连接层

    nn.Flatten(),
    #所以在最后变成的维数是和最后一个池化层所相关联的。
    nn.Linear(16*5*5,120),nn.Sigmoid(),
    #这个84是随便进行降的。
    nn.Linear(120,84),nn.Sigmoid(),
    nn.Linear(84,10)
)
#规定生成的X数据是四维数据。返回的是均匀分布。
#对数据集的测试
'''
X = torch.rand(size=(1,1,28,28),dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape :\t',X.shape)
'''
batch_size=256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
#print(train_iter.size())
'''
这个数据集将被分成235个批次，其中234个都是256，第235是96
'''
print(len(train_iter))

#这个就是对测试集数据来进行判断的
def evaluate_accuracy_gpu(net,data_iter,device=None):#@save
    '''使用gpu来进行计算'''

    if isinstance(net,nn.Module):
        net.eval()
        if not device:
            device=next(iter(net.parameters())).device

    #正确预测的数量
    metric = d2l.Accumulator(2)#累加器

    with torch.no_grad():

        for X,y in data_iter:

            if isinstance(X,list):

                X=[x.to(device) for x in X]

            else :
                X=X.to(device)

            #这个y还在for循环里面的
            y=y.to(device)

            #本来设置的累加器就是2个，现在 在这里面添加两个数字
            metric.add(d2l.accuracy(net(X),y),y.numel())

    return metric[0]/metric[1]


#@save
def train_ch6(net,train_iter,test_iter,num_epochs,lr,device):
    
    def init_weights(m):
        #卷积层还是全连接层都是有参数的，只有池化层是没有参数的。所以在卷积层和全连接的参数都是需要进行初始化的
        if type(m) == nn.Linear or type(m)==nn.Conv2d:
            #使用的初始化方式，
            nn.init.xavier_uniform_(m.weight)

    #在网络中应用自己初始化的参数
    net.apply(init_weights)

    print('training on',device)

    #将LeNet-5布置到GPU上面
    net.to(device)

    #网络中的参数,优化器使用的是随机梯度下降法
    optimizer = torch.optim.SGD(net.parameters(),lr=lr)

    #使用的是交叉熵函数作为误差
    loss =nn.CrossEntropyLoss()

    #
    #画图使用的函数
    # animator = d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],
    #                         legend=['train loss','train acc','test acc'])

    #时间，一共有235个批量，每一个批量大小为256。234*256=599904 再加上最后一个批量的大小是96
    timer,num_batchs =d2l.Timer(),len(train_iter)

    for epoch in range(num_epochs):
        #训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)#设置累加器的值为3，其中是3个0.0

        #开始训练网络
        net.train()

        #这里跑的都是训练集
        #enumerate() 这个函数只不过是加了一个索引，就是i
        '''
        在遍历train_iter这个的时候，将分别从0，1，2，3一直到234，这个是i的值
        这个内部的for循环将会进行235次循环
        '''
        for i,(X,y) in enumerate(train_iter):
            #开始计时
            timer.start()
            optimizer.zero_grad()

            X,y=X.to(device),y.to(device)

            y_hat=net(X)

            l=loss(y_hat,y)#默认求出来的是mean之后的损失大小

            l.backward()

            #更新参数
            optimizer.step()

            #shape[0]代表的是其批次的数量
            with torch.no_grad():
                #错误率乘上所有的数据就的出来了错误的个数
                '''
                然后在下面使用错误的个数再次除以总共的个数，就的出来了错误的概率
                '''
                metric.add(l*X.shape[0],d2l.accuracy(y_hat,y),X.shape[0])

            timer.stop()
            #
            train_l=metric[0]/metric[2]#代表的是每一个图片的损失大小
            train_acc=metric[1]/metric[2]#代表的是每一个epoch之后的准确度
            '''
            i的取值将为0，1，2，3，4--234，这将会是一个batch
            234=235-1
            下面这个if是为了最后一个循环所写的，也就是当批量大小是96的时候，会进入到下面
            只要是小数//大数，都将会是0
            '''

            '''
            i=0,1,2,3,,,,234
            num_bacths=235
            235//5=47
            0%47
            '''
            # if(i+1)%(num_batchs//5)==0 or i==num_batchs-1:
            #     animator.add(epoch+(i+1)/num_batchs,
            #                  #在这里为None是因为每一个批次的数目是不会发生改变的
            #                  (train_l,train_acc,None))

        '''
        “ // ” 表示整数除法，返回整数 比如 7/3 结果为2
        “ / ” 表示浮点数除法，返回浮点数 （即小数） 比如 8/2 结果为4.0
        “ %” 表示取余数 比如7/4 结果为3
        '''


        # 这里跑的都是训练集

        #这里跑的都是测试集，因为模型是在这里面跑出来的，所以还是需要在这里进行。跟随这模型，来进行每一次都迭代这个测试集，来看看每一次测试出来的结果都是如何的。
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        #animator.add(epoch + 1, (None, None, test_acc))

    print(f'loss{train_l:.3f},train acc{test_acc:.3f},test acc{test_acc:.3f}')#输出最后的结果

    #metric[2]总样本数 这个是时间
    print(f'{metric[2]*num_epochs/timer.sum():.1f} example/sec,on {str(device)}')
    #时间记录60000条数据的训练时间



lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().
    如果存在，则返回gpu（i），否则返回cpu（）。
    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')