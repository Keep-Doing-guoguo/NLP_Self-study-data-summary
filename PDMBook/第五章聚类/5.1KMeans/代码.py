import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
date = pd.read_csv('电信话单.csv')
fColumns = [
'工作日上班电话时长',
    '工作日下班电话时长',
    '周末电话时长', '国际电话时长',
    '总电话时长', '平均每次通话时长'
]
plt.figure()
#绘制散点图
axes = scatter_matrix(date[fColumns],diagonal='hist')



#计算相关系数矩阵
dataCorr = date[fColumns].corr()
fColumns = [
    '工作日上班电话时长', '工作日下班电话时长',
    '周末电话时长', '国际电话时长', '平均每次通话时长'
]

pca_2 = PCA(n_components=2)
date_pca_2 = pd.DataFrame(pca_2.fit_transform(date[fColumns]))
# plt.scatter(date_pca_2[0],date_pca_2[1])
#plt.show()

kmModel = KMeans(n_clusters=3)
kmModel = kmModel.fit(date_pca_2)
#Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)
# pTarget = kmModel.predict(date[fColumns])

#将降维后的数据，分类之后进行可视化显示。
# plt.figure()
# plt.scatter(date_pca_2[0],date_pca_2[1],c=pTarget)
# plt.show()
