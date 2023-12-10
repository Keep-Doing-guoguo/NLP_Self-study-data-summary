import pandas as pd
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder


data = pd.read_csv('华南地区.csv')

data['注册时长_cut'] = pd.qcut(data['注册时长'],10)
a = data['注册时长_cut'].unique()
data['营收收入_cut'] = pd.qcut(data['营收收入'],10)
data['成本_cut'] = pd.qcut(data['成本'],10)

b = data.groupby('注册时长_cut')['ID'].count()
c = data.groupby('营收收入_cut')['ID'].count()
d = data.groupby('成本_cut')['ID'].count()

#特征变量
x = data[['注册时长', '营收收入', '成本']]
#目标变量
y = data['是否续约']
gaussianNB = GaussianNB()
e = cross_val_score(
    gaussianNB,
    x, y, cv=3,
).mean()

cutX = data[[
    '注册时长_cut',
    '营收收入_cut',
    '成本_cut'
]]
onehot = OneHotEncoder()
cudXdata = onehot.fit_transform(cutX)
cudXdata = cudXdata.toarray()

bernou = BernoulliNB()
cvs = cross_val_score(bernou,cudXdata,y,cv=3)

print()