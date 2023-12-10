from sklearn.datasets import load_digits
import numpy as np
#是一个用于二值化数据的类，它可以将数值特征二值化为0和1。
from sklearn.preprocessing import Binarizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
#加载手写数字案例数据
mnist = load_digits()
x = mnist.data
y = mnist.target
#拿出第一个样本出来演示
# x = x.reshape(-1,8,8)
bina = Binarizer()
binadata = bina.fit_transform(x)

gaussianNB = GaussianNB()
cvs = cross_val_score(gaussianNB,x,y,cv=3)
a = cvs.mean()

print()


print()
