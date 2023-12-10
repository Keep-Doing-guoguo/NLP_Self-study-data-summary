import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import  MultinomialNB
from sklearn.model_selection import cross_val_score

path = '/Users/zhangguowen/Library/Mobile Documents/com~apple~CloudDocs/pythonProject1/PDMBook/第三章分类模型/3.5贝叶斯分类/多项式贝叶斯.xlsx'
data = pd.read_excel(path)

#进行中文分词
fileContents = []
for index,row in data.iterrows():
    fileContent = row['fileContent']
    segs = jieba.cut(fileContent)
    fileContents.append(''.join(segs))
data['file_content_cut'] = fileContents
#文本向量化
countVect = CountVectorizer(
    min_df=0,
    token_pattern=r'\b\w+\b'#正则表达式
)
textVector = countVect.fit_transform(data['file_content_cut'])
MNBModel = MultinomialNB()
#进行K折交叉验证
cvs = cross_val_score(MNBModel,textVector,data['class'],cv=3)
print(cvs.mean())


print()