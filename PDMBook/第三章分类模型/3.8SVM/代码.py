#verbose用来控制输出详细程度

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
winedata = pd.read_csv('SVM.csv')
a = winedata['label']
x = winedata[[
'Alcohol', 'Malic acid', 'Ash',
    'Alcalinity of ash','Magnesium',
    'Total phenols', 'Flavanoids',
    'Nonflavanoid phenols',
    'Proanthocyanins', 'Color intensitys',
    'Hue', 'OD280/OD315 of diluted wines',
    'Proline'
]]
y = winedata['label']



print()
svc = SVC()
params = dict(
    kernel = ['poly'],
    degree = [5,6,7,8]
)
grid = GridSearchCV(
    svc,params,cv=3,verbose=1,n_jobs=5,return_train_score=True
)
b = grid.fit(x,y)
print('最好的得分是: %f' % grid.best_score_)
print('最好的参数是:')
for key in grid.best_params_.keys():
    print('%s=%s'%(key, grid.best_params_[key]))