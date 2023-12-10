import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score

path = '/Users/zhangguowen/Library/Mobile Documents/com~apple~CloudDocs/pythonProject1/PDMBook/第三章分类模型/3.5贝叶斯分类/议案投票.csv'
data =pd.read_csv(path)
a = data.shape
b = data.info()
c = data.isnull().sum()
# print(b)
# print('===========')
# print(c)
#填充缺失值
data = data.fillna('None')
#所有的议题作为特征
features = [
    'Campaign Finance Overhaul',
    'Unemployment and Tax Benefits',
    'Fiscal 2003 Budget Resolution',
    'Permanent Tax Cuts', 'Food Stamps',
    'Nuclear Waste', 'Fiscal 2003 Defense Authorization',
    'Abortions Overseas', 'Defense Authorization Recommitment',
    'Welfare Renewal', 'Estate Tax Repeal',
    'Married Couples Tax Relief', 'Late Term Abortion Ban',
    'Homeland Sec/Union Memb', 'Homeland Sec/Civil Service Emp',
    'Homeland Sec/Whistleblower Protections', 'Andean Trade',
    'Abortion Service Refusals', 'Medical Malpractice Awards',
    'Military Support for UN Resolution'
]
#新建独热编码器
oneHotEncoder = OneHotEncoder()
#训练独热编码器
oneHotEncoder.fit(data[features])
#转换数据
oneHotData = oneHotEncoder.transform(data[features])
#print(type(oneHotData))
BNBModel = BernoulliNB()
cvs = cross_val_score(BNBModel,oneHotData,data['Party'],cv=10)
d = cvs.mean()

BNBModel = BernoulliNB()
#使用所有数据训练模型
BNBModel.fit(oneHotData,data['Party'])
#对所有的数据进行预测
data['Predict Party'] = BNBModel.predict(oneHotData)
confusion_matrix(
    data['Party'],
    data['Predict Party'],
    labels=['D', 'R']
)
e = pd.crosstab(data['Party'],data['Predict Party'])
print(e)

f = BNBModel.score(oneHotData,data['Party'])
print(f)



print()



def learn_oneHot():

    data = pd.DataFrame({
        'color':['red','blue','black','red']
    })
    data = pd.DataFrame({
        'gender':['男','女','男','女'],
        'location':['上海','北京','广州','上海']
    })
    print(data)
    data = oneHotEncoder.fit_transform(data)
    print(data)#在这里只返回了带1的坐标位置。

    from sklearn.preprocessing import OneHotEncoder
    import numpy as np

    # 创建示例数据
    data = np.array([['男', '上海'], ['女', '北京'], ['男', '广州'], ['女', '上海']])

    # 创建OneHotEncoder对象
    encoder = OneHotEncoder()

    # 将数据进行one-hot编码
    encoder.fit(data)
    result = encoder.transform(data).toarray()

    # 输出编码结果
    print(result)


