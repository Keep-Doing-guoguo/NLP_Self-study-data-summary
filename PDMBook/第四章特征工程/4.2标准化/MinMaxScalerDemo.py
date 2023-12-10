import matplotlib.pyplot as plt
import pandas as  pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer,f1_score,recall_score,accuracy_score,precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


data = pd.read_csv('华南地区.csv')

#特征变量
x = data[['注册时长','营收收入','成本']]
#目标变量
y = data['是否续约']

#生成标准化对象
scaler = MinMaxScaler()
scalerX = scaler.fit_transform(x)

#用来保存KNN模型的neighbors的个数的
k = []
accuracy_means = []
precision_means = []
recall_means = []
f1_means = []
for i in range(2,30):
    k.append(i)
    kNNModel = KNeighborsClassifier(n_neighbors=i)
    cvs = cross_val_score(kNNModel,x,y,cv=10,scoring=make_scorer(accuracy_score))
    accuracy_means.append(cvs.mean())

    cvs = cross_val_score(kNNModel,x,y,cv=10,scoring=make_scorer(precision_score,pos_label = '续约'))
    precision_means.append(cvs.mean())

    cvs = cross_val_score(kNNModel,x,y,cv=10,scoring=make_scorer(recall_score,pos_label='续约'))
    recall_means.append(cvs.mean())

    cvs = cross_val_score(kNNModel,x,y,cv=10,scoring=make_scorer(f1_score,pos_label='续约'))
    f1_means.append(cvs.mean())

#生成参数对应的模型评分
scores = pd.DataFrame({
    'precision': precision_means,
    'accuracy': accuracy_means,
    'recall': recall_means,
    'f1': f1_means
})

# plt.figure()
# plt.plot(k,precision_means)
# plt.show()