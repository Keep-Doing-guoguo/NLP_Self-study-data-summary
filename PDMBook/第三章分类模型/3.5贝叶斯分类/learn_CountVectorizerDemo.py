import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer



#增加词的权重，方便分词
jieba.add_word('书',freq=1000000)
jieba.add_word('篮球',freq=100000)
jieba.add_word('乒乓球',freq=100000)

#读取案例数据
data = pd.read_csv('CountVectorizerDemo.csv')
a = data.text
data['text_cut'] = data.text.apply(
    lambda t:' '.join(jieba.cut(t))
)
#新建文本计数向量化器
counter = CountVectorizer(
    min_df=0,token_pattern=r'\b\w+\b'
)
#训练文本计数向量化器
counter.fit(data['text_cut'])
#获取特征词字典
vocabulary = counter.vocabulary_
#按照顺序对词字典进行排序显示
b = sorted(vocabulary.items(),key=lambda x: x[1])
#把文本转换为文本向量
textVector = counter.transform(
    data['text_cut']
)
#输出显示文本向量
c = textVector.toarray()

print()