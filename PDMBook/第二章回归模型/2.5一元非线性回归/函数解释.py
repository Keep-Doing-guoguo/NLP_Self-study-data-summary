import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# 创建一个二维特征数组
X = np.array([[1, 2], [3, 4], [5, 6]])

# 创建一个PolynomialFeatures对象，将二维特征转换为二次多项式特征
poly = PolynomialFeatures(degree=2)

# 将二维特征转换为二次多项式特征
X_poly = poly.fit_transform(X)

# 输出转换后的特征数组
print(X_poly)
'''
可以看到，原始的二维特征数组被转换为了一个包含6列的二次多项式特征数组。
其中，第一列为常数项，第二列和第三列为原始特征，第四列为第一列特征的平方，
第五列为第一列特征与第二列特征的乘积，第六列为第二列特征的平方。
'''