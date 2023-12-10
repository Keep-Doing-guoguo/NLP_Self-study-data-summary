import pandas

data = pandas.read_csv(
    '华南地区.csv',
    engine='python', encoding='utf8'
)

# 特征变量
x = data[['注册时长', '营收收入', '成本']]
# 目标变量
y = data['是否续约']

from sklearn.preprocessing import Normalizer

# 生成标准化对象
scaler = Normalizer()
# 训练标准化对象
scaler.fit(x)
# 把数据转换为标准化数据
scalerX = scaler.transform(x)

x = scaler.transform(x)

from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# 用来保存KNN模型的邻居个数
ks = []
# 用来保存准确率
accuracy_means = []
# 用来保存精确率
precision_means = []
# 用来保存召回率
recall_means = []
# 用来保存f1值
f1_means = []
# n_neighbors参数，从2到29，一个个尝试
for k in range(2, 30):
    # 把n_neighbors参数保存起来
    ks.append(k)
    # 改变KNN模型的参数n_neighbors为k
    knnModel = KNeighborsClassifier(n_neighbors=k)
    # 计算10折交叉验证的准确率
    accuracy_cvs = cross_val_score(
        knnModel,
        x, y, cv=10,
        scoring=make_scorer(accuracy_score)
    )
    # 将10折交叉验证的准确率的均值保存起来
    accuracy_means.append(accuracy_cvs.mean())
    # 计算10折交叉验证的精确率
    precision_cvs = cross_val_score(
        knnModel,
        x, y, cv=10,
        scoring=make_scorer(
            precision_score,
            pos_label="续约"
        )
    )
    # 将10折交叉验证的精确率的均值保存起来
    precision_means.append(precision_cvs.mean())
    # 计算10折交叉验证的召回率
    recall_cvs = cross_val_score(
        knnModel,
        x, y, cv=10,
        scoring=make_scorer(
            recall_score,
            pos_label="续约"
        )
    )
    # 将10折交叉验证的召回率的均值保存起来
    recall_means.append(recall_cvs.mean())
    # 计算10折交叉验证的f1值
    f1_cvs = cross_val_score(
        knnModel,
        x, y, cv=10,
        scoring=make_scorer(
            f1_score,
            pos_label="续约"
        )
    )
    # 将10折交叉验证的f1值的均值保存起来
    f1_means.append(f1_cvs.mean())

# 生成参数对应的模型评分
scores = pandas.DataFrame({
    'k': ks,
    'precision': precision_means,
    'accuracy': accuracy_means,
    'recall': recall_means,
    'f1': f1_means
})

# 绘制不同参数对应的评分的折线图
scores.plot(
    x='k',
    y=['accuracy', 'precision', 'recall', 'f1']
)
