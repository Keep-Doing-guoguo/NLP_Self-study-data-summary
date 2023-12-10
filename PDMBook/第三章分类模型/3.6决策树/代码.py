import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO
from scipy.sparse import hstack
from sklearn.model_selection import cross_val_score

data = pd.read_csv('决策树.csv')
#需要进行OneHot编码处理
OnetHot = OneHotEncoder(drop='first')
features = ['性别','父母鼓励']
dataOne = OnetHot.fit_transform(data[features])
dataOne = dataOne.toarray()

#==================================#
#第一步：数据转换格式
data_ = np.array(data[['父母收入','IQ']])
#第二步：直接进行拼接
data_1 = np.hstack((data_,dataOne))
#第三步：再将数据转换为pandas
x = pd.DataFrame(data_1)
x = x.rename(columns={0:'父母收入',1:'IQ',2:'性别',3:'父母鼓励'})
print(x.head())
#==================================#

y = data['升学计划']
#设置树的深度为3，最大叶子结点为7.
dtModel = DecisionTreeClassifier(
    max_depth=3,
    max_leaf_nodes=7
)
cvs = cross_val_score(dtModel,x,y,cv=10)
a = cvs.mean()

dtModel = DecisionTreeClassifier()
dtModel.fit(x,y)
b = dtModel.score(x,y)

dtModel = DecisionTreeClassifier()
#网格搜索，寻找最优参数
#将两个列表转换为字典类型的数据
paramGrid = dict(
    max_depth = [1,2,3,4,5],
    max_leaf_nodes = [3,5,6,7,8]
)
grid = GridSearchCV(
    dtModel,paramGrid,cv=10,return_train_score=True
)
grid = grid.fit(x,y)

print('最好的得分是：%f' % grid.best_score_)
print('最好的参数是：')
print(grid.best_params_)
for key in grid.best_params_.keys():
    print('%s=%s' % (key, grid.best_params_[key]))

print()
