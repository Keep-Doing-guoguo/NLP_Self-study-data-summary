import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

path = '/Users/zhangguowen/Library/Mobile Documents/com~apple~CloudDocs/pythonProject1/PDMBook/第三章分类模型/3.5贝叶斯分类/高斯贝叶斯.csv'
data = pd.read_csv(path)

features = ['注册时长','营收收入','成本']

#高斯贝叶斯
gauss = GaussianNB()
#进行k折交叉验证
csv = cross_val_score(gauss,data[features],data['是否续约'],cv=10)
a = csv.mean()

gauss = GaussianNB()
#使用所有的数据训练模型
gauss.fit(data[features],data['是否续约'])
#对所有的数据进行预测
data['预测是否续约'] = gauss.predict(data[features])
b = confusion_matrix(data['是否续约'],data['预测是否续约'],labels=['不续约','续约'])

c = gauss.score(data[features],data['是否续约'])

print()