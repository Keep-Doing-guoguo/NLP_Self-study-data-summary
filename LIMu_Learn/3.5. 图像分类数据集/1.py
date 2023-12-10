
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils import data

#(图像变换)
#在 torch.transforms 中有大量的数据变换类
from torchvision import transforms

from d2l import torch as d2l

d2l.use_svg_display()


#3.5.1. 读取数据集
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0到1之间
trans = transforms.ToTensor()

#中60000张训练图像加10000张测试图像，图像大小为28x28，单通道，共分10个类
mnist_train = torchvision.datasets.FashionMNIST(
    ## 将数据保存在本地什么位置
    ## 我们希望数据用于训练集，其中6万张图片用作训练数据，1万张图片用于测试数据
    # 如果目录下没有文件，则自动下载
    ## 我们将数据转为Tensor类型
    #非依次对应
    root="../data", train=True, transform=trans, download=True)

mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
#训练集有60000，测试集有10000条。
print(len(mnist_train), len(mnist_test))

#单通道
#print(mnist_train[0][0].shape)

#以下函数用于在数字标签索引及其文本名称之间进行转换。
def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

#可视化操作
#输入的参数分别是图像 sacle每张图片的大小
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)#9*1.5 2*1.5
    #print(imgs)
    #创建一个大小为2行9列的画布
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)

    #print(axes)
    axes = axes.flatten()
    #展平之后变成了1行18列
    #print(axes)
    '''
    zip(axes, imgs)
    将axes与imgs中的元素进行相互对应
    zip过后变成了一个列表
    
    enumerate：对其进行循环得到的是0，1和对应的（ax，img）
    
    输出的ax应该是（0，0）（0，1）----（0，17）
    '''
    for i, (ax, img) in enumerate(zip(axes, imgs)):

        if torch.is_tensor(img):
            # 图片张量 画出来的是热图
            ax.imshow(img.numpy())
            #print(img)
        else:
            # PIL图片
            ax.imshow(img)

        #ax对应的就是大图中的小图。将x，y坐标进行隐藏
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        if titles:
            ax.set_title(titles[i])
    d2l.plt.show()
    return axes

#在这里并未进行顺序打乱，只输出了一次的batch_size的大小
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))

#y的分布是0-9进行分布的
#print(y)
#X的每一个大小都是28*28，18代表的是每一个bacth-size
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));

#3.5.2. 读取小批量
batch_size = 256

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4

#对所有的数据进行分批
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())

timer = d2l.Timer()
#在这里是测试遍历整个数据集所需要的时间是多少。
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')


#3.5.3. 整合所有组件
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]#归一化操作
    if resize:
        #对归一化之后的操作，将其在其4周进行0填充
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)

    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    #加上break的意思是只输出1次。
    break