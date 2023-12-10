import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,\
    recall_score,f1_score,make_scorer
#导入数据
path = '/Users/zhangguowen/Library/Mobile Documents/com~apple~CloudDocs/pythonProject1/PDMBook/第三章分类模型/3.4KNN/华南地区.csv'
data = pd.read_csv(path)
a = data.shape#shape是属性
huabei_path = '/Users/zhangguowen/Library/Mobile Documents/com~apple~CloudDocs/pythonProject1/PDMBook/第三章分类模型/3.4KNN/华北地区.csv'
huabei_data = pd.read_csv(huabei_path)

#特征变量
#x = data[['注册时长'],['营收收入'],['成本']]
x = data[['注册时长', '营收收入', '成本']]
#目标变量
y = data['是否续约']

#把数据分为训练集和测试集
x_trian,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.3)
#新建一个KNN模型，设置个数为3.
knnModel = KNeighborsClassifier(n_neighbors=3)
#使用训练集训练KNN模型
knnModel.fit(x_trian,y_train)
knnModel.score(x_test,y_test)

#预测测试集数据集的目标变量
y_test_predict = knnModel.predict(x_test)
b = confusion_matrix(y_test,y_test_predict,labels=['续约','不续约'])

#准确率
c = accuracy_score(y_test,y_test_predict)

#精确率
d = precision_score(y_test,y_test_predict,pos_label='续约')
#召回率
e = recall_score(y_test,y_test_predict,pos_label='续约')
#f1score
f = f1_score(y_test,y_test_predict,pos_label='续约')

#进行k折交叉验证
knnModel = KNeighborsClassifier(n_neighbors=3)
cvs = cross_val_score(knnModel,x,y,cv=10)
g = cvs
h = cvs.mean()


#
ks = []
accuracy_means = []
precision_means = []
recall_means = []
f1_means = []
for k in range(2,30):
    #把n_neighbors参数保存下来
    ks.append(k)

    knnModel = KNeighborsClassifier(n_neighbors=k)
    #计算10折交叉验证的准确率
    cvs = cross_val_score(knnModel,x,y,cv=10,scoring=make_scorer(accuracy_score))
    #cvs = cross_val_score(knnModel,x,y,cv=10,scoring=accuracy_means)
    accuracy_means.append(cvs.mean())

    #计算10折交叉验证的精确度
    cvs = cross_val_score(knnModel,x,y,cv=10,scoring=make_scorer(precision_score,pos_label='续约'))
    precision_means.append(cvs.mean())
    #计算10折交叉验证的召回率
    cvs = cross_val_score(knnModel,x,y,cv=10,scoring=make_scorer(recall_score,pos_label='续约'))
    recall_means.append(cvs.mean())
    #计算10折交叉验证的f1_score
    cvs = cross_val_score(knnModel,x,y,cv=10,scoring=make_scorer(f1_score,pos_label='续约'))
    f1_means.append(cvs.mean())

#生成各种参数对应的模型评分
scores = pd.DataFrame({
    'k':ks,
    'precision':precision_means,
    'accuracy':accuracy_means,
    'recall':recall_means,
    'f1':f1_means
})
#绘制不同参数对应的评分折线图
scores.plot(
    x = 'k',
    y = ['precision','accuracy','recall','f1']
)
# plt.show()
#使用最佳参数，这个可以根据图片可以得知。来进行建模。
knnModel = KNeighborsClassifier(n_neighbors=17)
#使用所有的训练样本训练模型
knnModel.fit(x,y)
#对未知的目标数据进行预测
data = huabei_data[['注册时长', '营收收入', '成本']]
huabei_data['预测续约'] = knnModel.predict(data)

print()