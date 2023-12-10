import pandas

data = pandas.read_csv(
    '层次聚类.csv',
    encoding='utf8', engine='python'
)

import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
#设置中文字体
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

fColumns = [
    '工作日上班时间人均停留时间',
    '凌晨人均停留时间',
    '周末人均停留时间',
    '日均人流量'
]

from sklearn.preprocessing import scale
#因为人流量和时间属于不同的计量单位，
#因此需要对这份数据需要进行标准化
scaleData = pandas.DataFrame(
    scale(data[fColumns]), columns=fColumns
)

#绘制散点矩阵图
axes = scatter_matrix(
    scaleData, diagonal='hist'
)


from sklearn.decomposition import PCA

pca_2 = PCA(n_components=2)
data_pca_2 = pandas.DataFrame(
    pca_2.fit_transform(scaleData)
)
plt.scatter(
    data_pca_2[0],
    data_pca_2[1]
)


from sklearn.cluster import AgglomerativeClustering
#进行层次聚类，并预测样本的分组
agglomerativeClustering = AgglomerativeClustering(n_clusters=3)
pTarget = agglomerativeClustering.fit_predict(scaleData)

plt.figure()
plt.scatter(
    data_pca_2[0],
    data_pca_2[1],
    c=pTarget
)
plt.show()
import scipy.cluster.hierarchy as hcluster
#构建层次聚类树
linkage = hcluster.linkage(
    scaleData,
    method='centroid'
)
#绘制层次聚类图形
plt.figure()
hcluster.dendrogram(
    linkage,
    leaf_font_size=10.
)
#计算层次聚类结果
_pTarget = hcluster.fcluster(
    linkage, 3,
    criterion='maxclust'
)



import seaborn as sns
from pandas.plotting import parallel_coordinates

fColumns = [
    '工作日上班时间人均停留时间',
    '凌晨人均停留时间',
    '周末人均停留时间',
    '日均人流量',
    '类型'
]

data['类型'] = pTarget

plt.figure()
ax = parallel_coordinates(
    data[fColumns], '类型',
    color=sns.color_palette(),
)



from sklearn.decomposition import PCA

pca_2 = PCA(n_components=2)
data_pca_2 = pandas.DataFrame(
    pca_2.fit_transform(scaleData)
)
plt.scatter(
    data_pca_2[0],
    data_pca_2[1]
)
