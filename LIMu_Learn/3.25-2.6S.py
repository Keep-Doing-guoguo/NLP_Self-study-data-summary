#%matplotlib inline

#首先，我们导入必要的软件包。

#大数定律（law of large numbers）告诉我们： 随着投掷次数的增加，这个估计值会越来越接近真实的潜在概率。
import matplotlib.pyplot as plt
import torch
from torch.distributions import multinomial
from d2l import torch as d2l

#创建一个全部是1的向量，然后用这个向量来除以6，得到的值就是[0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667]
fair_probs = torch.ones([6]) / 6
print(fair_probs)

#随机选择样本 这个是只不过是选择了1边,在这里得到了1一个6行1列的向量
print(multinomial.Multinomial(1, fair_probs).sample())

#在下面我要选择10边，对于这个样本 在此得到了一个6行1列的向量
print(multinomial.Multinomial(10, fair_probs).sample())

# 将结果存储为32位浮点数以进行除法 在此得到了一个6行1列的向量
counts = multinomial.Multinomial(1000, fair_probs).sample()

#因为是1000次，所以将每一次都除1000，得到数字趋近于1/6
print(counts / 1000 ) # 相对频率作为估计值

#画图进行显示结果
#每条实线对应于骰子的6个值中的一个，并给出骰子在每组实验后出现值的估计概率。 当我们通过更多的实验获得更多的数据时，这条实体曲线向真实概率收敛。
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
#进行三次循环，可以得到1，2，3，4，5，6随着实验进行下去所出现的概率
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 #画出来每一条的线
                 label=("P(die=" + str(i + 1) + ")"))
#将0.167这条线画出来，可以看到所出现的概率和这个数字有多接近。
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
#设置横轴的名称
d2l.plt.gca().set_xlabel('Groups of experiments')
#设置竖轴的名称
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
plt.show()