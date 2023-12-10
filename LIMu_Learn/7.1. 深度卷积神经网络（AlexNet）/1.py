import torch
from torch import nn
from d2l import torch as d2l

#定义AlexNet网络总共是8层，5个卷积层，3个全连接层
#来自的图片大小是3*224*224
net = nn.Sequential(
        #在这里使用的是11*11的更大窗口来捕捉对象
        #同时，步幅为4，以减少输出的高度和宽度
        #另外，输出通道数目远大于LeNet


        nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1),nn.ReLU(),
        nn.MaxPool2d(kernel_size=3,stride=2),

        #使用2填充来保持图像的大小不变
        #池化层保持不变
        nn.Conv2d(96,256,kernel_size=5,padding=2),nn.ReLU(),
        nn.MaxPool2d(kernel_size=3,stride=2),

        #增加通道数，
        nn.Conv2d(256,384,kernel_size=3,padding=1),nn.ReLU(),
        nn.Conv2d(384,384,kernel_size=3,padding=1),nn.ReLU(),
        nn.Conv2d(384,256,kernel_size=3,padding=1),nn.ReLU(),

        nn.MaxPool2d(kernel_size=3,stride=2),
        nn.Flatten(),
        #z这里，全连接层的输出数量是LeNet的好几倍。使用dropout来减轻过拟合。LeNet的全连接层是16*5*5=400
        nn.Linear(6400,4096),nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096,4096),nn.ReLU(),
        nn.Dropout(p=0.5),
        #zui最后是输出层。由于这里是使用的Fashion——MINST，所以用的类别数为10，
        nn.Linear(4096,10)
    )



#查看得到数据集的样子

X=torch.randn(1,1,224,224)
for layer in net:
        X=layer(X)
        print(layer.__class__.__name__,'output shape:\t',X.shape)



#读取数据集
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

#训练AlexNet
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
