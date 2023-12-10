import jieba


a = '我喜欢打高尔夫'
aa = jieba.cut(a)
b = list(jieba.cut(a))

#给高尔夫这个词，设置一个较高的权限
jieba.add_word('高尔夫',freq=1000000)
c = jieba.cut(a)
d = list(jieba.cut(a))
print()