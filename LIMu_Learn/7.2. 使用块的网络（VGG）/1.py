#定义VGG块
import torch
from torch import nn
from d2l import torch as d2l

#分别依次是卷积层的个数，输入的通道数，输出的通道数
#使用多个卷积层，然后再加上非线性激活函数，再加上
def vgg_block(num_convs,in_channels,out_channerls):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels,out_channerls,kernel_size=3,padding=1))
        layers.append(nn.ReLU())
        in_channels=out_channerls

    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
'''
这里代表的是卷积层的个数，然后再加上一个池化层，将高宽减半。
Sequential(
  (0): Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU()
  (2): Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (3): ReLU()
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
'''
#print(vgg_block(2,224,224))

def vgg(conv_arch):
    #卷积块的个数应该设置多少个
    con_blks=[]
    in_channels=1
    #卷积层部分
    '''
    每经历一次vgg_block都会将长宽减半，再加上一层的卷积层.输出通道是自己进行设计的，每次都是再进行增加的
    '''
    for (num_convs,out_channerls) in conv_arch:
        vgg_block1=vgg_block(num_convs,in_channels,out_channerls)
        con_blks.append(vgg_block1)
        in_channels=out_channerls


    return nn.Sequential(
        *con_blks,
        #然后再进行展平操作
        nn.Flatten(),
        #进入到全链接层部分
        nn.Linear(out_channerls*7*7,4096),nn.ReLU(),nn.Dropout(0.5),
        nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(0.5),
        nn.Linear(4096,10)
    )

conv_arch=((1,64),(1,128),(2,256),(2,512),(2,512))
'''
其中前面的1，代表的是一共有几个卷积层
像这个((1,64),(1,128),(2,256),(2,512),(2,512))，其意思就是
卷积层为1 
Sequential(
  (0): Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU()
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
高宽减半为224/2=112
通道数由1-64

卷积层为1
Sequential(
  (0): Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU()
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
高宽减半为112/2=56
通道数64-128

卷积层为2
Sequential(
  (0): Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU()
  (2): Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (3): ReLU()
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
高宽减半为56/2=28
通道数128-256

卷积层为2
Sequential(
  (0): Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU()
  (2): Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (3): ReLU()
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
高宽减半为28/2=14
通道数256-512

卷积层为2
Sequential(
  (0): Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU()
  (2): Conv2d(224, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (3): ReLU()
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
高宽减半为14/2=7
通道数512-512
'''
net = vgg(conv_arch)
X=torch.randn(size=(1,1,224,224))
for blk in net:
    X=blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

'''
1+1+2+2+2+3个全连接层
'''
ratio = 4
#我们构建了一个通道数较少的网络
#[(1, 16), (1, 32), (2, 64), (2, 128), (2, 128)]
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
print(small_conv_arch)
net = vgg(small_conv_arch)