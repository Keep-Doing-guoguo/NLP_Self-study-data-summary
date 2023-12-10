#数据操作
#1.入门
# 具有一个轴的张量对应数学上的向量（vector）；
# 具有两个轴的张量对应数学上的矩阵（matrix）
# 具有两个轴以上的张量没有特殊的数学名称。
import torch
import numpy
x = torch.arange(12)
print(x)

#访问张量（沿每个轴的长度）的形状
print(x.shape)
print(x.numel())

#要想改变一个张量的形状而不改变元素数量和元素值，可以调用reshape函数。
# 将其改变为3*4的矩阵，但是其数据内容并不会发生改变
# 即我们可以用x.reshape(-1,4)或x.reshape(3,-1)来取代x.reshape(3,4)。
X = x.reshape(3, 4)
print(X)

#有时，我们希望使用全0、全1、其他常量
XX=torch.zeros((2, 3, 4))
print(XX)

#我们可以创建一个形状为(2,3,4)的张量
XX=torch.ones((2, 3, 4))
print(XX)

#我们通常会随机初始化参数的值
print(torch.randn(3, 4))

#所需张量中的每个元素赋予确定值。赋值操作,一个矩阵
print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))

#2运算符
#在这些数据上执行数学运算
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y)
print( x - y)
print(x * y)
print(x / y)
print(x ** y)# **运算符是求幂运算

#求幂这样的一元运算符
print(torch.exp(x))

#执行线性代数运算，包括向量点积和矩阵乘法

#多个张量连结（concatenate）在一起， 把它们端对端地叠起来形成一个更大的张量.
# 创建一个向量  类型是浮点型的 然后将其改为3*4的矩阵
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
# 创建一个3*4的向量
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0))# 按照行 添加到下面
print(torch.cat((X, Y), dim=1))#按照列 添加到左边

#通过逻辑运算符构建二元张量。
# 以X == Y为例： 对于每个位置，如果X和Y在该位置相等，则新张量中相应项的值为1。
print(X == Y)

#对张量中的所有元素进行求和，会产生一个单元素张量。
print(X.sum())

#3广播机制
#在某些情况下，即使形状不同，
# 我们仍然可以通过调用 广播机制（broadcasting mechanism）来执行按元素操作。
# 将张量改变为3行1列
a = torch.arange(3).reshape((3, 1))
#将张量改变为 1行2列
b = torch.arange(2).reshape((1, 2))
print(a)
print(b)

#由于a和b分别是3*1和1*2矩阵，如果让它们相加，它们的形状不匹配。
#矩阵a将复制列， 矩阵b将复制行，然后再按元素相加。
print(a + b)

#4索引和切片 读取元素
#张量中的元素可以通过索引访问。
#第一个元素的索引是0，最后一个元素索引是-1；
# 可以指定范围以包含第一个元素和最后一个之前的元素。
# 其表示的是最后一行的元素
print(X[-1])
# 其表示的是第二行到最后一行 0是第一行 1是第二行 2是第三行
print(X[1:3])

#除读取外，我们还可以通过指定索引来将元素写入矩阵。
#如果加上了逗号，那么前面的就会表示为行，后面的就是列
#所以就是第二行 第三列的数字为9
X[1, 2] = 9
print(X)

#我们想为多个元素赋值相同的值，我们只需要索引所有元素，然后为它们赋值
#代表的是第一行到第二行的所有元素，后面的数字是属于不包含的状态
X[0:2, :] = 12
print(X)

#5节省内存
#运行一些操作可能会导致为新结果分配内存。
# 例如，如果我们用Y = X + Y，我们将取消引用Y指向的张量，而是指向新分配的内存处的张量。
#Python的id()函数演示了这一点， 它给我们提供了内存中引用对象的确切地址。
before = id(Y)
Y = Y + X#在这里相当于重新建了一个新的变量
print(id(Y) == before)

#执行原地操作非常简单。
#我们首先创建一个新的矩阵Z，其形状与另一个Y相同， 使用zeros_like来分配一个全的块。
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
#代表的是Z的所有的元素 这是原地操作
Z[:] = X + Y
print('id(Z):', id(Z))

#原地操作
before = id(X)
X += Y
id(X) == before

#6转换为其他py对象
#torch张量和numpy数组将共享它们的底层内存，
# 就地操作更改一个张量也会同时更改另一个张量。
A = X.numpy()
B = torch.tensor(A)
print(type(A), type(B))

a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))
#7小结
#张量（维数组） 就是矩阵 向量
