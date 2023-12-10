import pandas as pd
from sklearn.preprocessing import scale
from sklearn.decomposition import FactorAnalysis
import matplotlib
import matplotlib.pyplot as plt
fa = FactorAnalysis(n_components=2)#降至到2维
data = pd.read_csv('fa.csv')
datascale = scale(data)
fadata = fa.fit_transform(datascale)
#获取因子
loadingVSK = pd.DataFrame({
    'PA1':fa.components_[0],
    'PA2':fa.components_[1]
})
#把列名添加到数据框中
loadingVSK['colName'] = data.columns.values

fig = plt.figure()
#画出原点坐标轴
plt.axvline(x=0, linewidth=1)
plt.axhline(y=0, linewidth=1)
plt.scatter(
    loadingVSK['PA1'],
    loadingVSK['PA2']
)

plt.show()
print()
