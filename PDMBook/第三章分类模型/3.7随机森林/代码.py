import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from scipy.sparse import hstack
import numpy as np

path = '随机森林.csv'
data = pd.read_csv(path)
#需要进行OneHot处理的列
ontHot = ['性别','父母鼓励']
oneHoten = OneHotEncoder(drop='first')
dataOneHot = oneHoten.fit_transform(data[ontHot])

#使用numpy来合并数据
dataOneHot = dataOneHot.toarray()
x = np.hstack((dataOneHot,data[['父母收入','IQ']]))
y = data['升学计划']

rclassifier = RandomForestClassifier()
#网格搜索，寻找最优参数
paramGrid = dict(
    max_depth = [1,2,3,4,5],
    criterion = ['gini','entropy'],
    max_leaf_nodes = [3,5,6,7,8],
    n_estimators = [10,50,100,150,200]
)

gridSearchCV = GridSearchCV(
    rclassifier,paramGrid,
    cv=10,verbose=1,n_jobs=10,
    return_train_score=True
)
grid = gridSearchCV.fit(x,y)
print('最好的得分是：%f' % grid.best_score_)
print('最好的参数是：')
for key in grid.best_params_.keys():
    print(grid.best_params_[key])


print()