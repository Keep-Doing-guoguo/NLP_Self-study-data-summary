# -*- coding: utf-8 -*-

import pandas
import matplotlib
import matplotlib.pyplot as plt

data = pandas.read_csv(
    'movie.csv',
    encoding='utf8', 
    engine='python'
)

mainColor = (42/256, 87/256, 141/256, 1)
#通过字体路径和文字大小，
#生成字体属性，赋值给 font 变量
font = matplotlib.font_manager.FontProperties(
    fname='/Users/zhangguowen/Library/Mobile Documents/com~apple~CloudDocs/pythonProject1/PDMBook/SourceHanSansCN-Light.otf',
    size=30
)

fig = plt.figure()

plt.xlabel('累计票房', fontproperties=font)
plt.ylabel('豆瓣评分', fontproperties=font)
plt.xlim([0, data['累计票房'].max()*1.5])
plt.ylim([0, data['豆瓣评分'].max()*1.5])

#画线
plt.scatter(
    data['累计票房'], data['豆瓣评分'], 
    alpha=0.5, s=200, marker="o", 
    facecolors='none', 
    edgecolors=mainColor, 
    linewidths=5
)

data.apply(
    lambda row: plt.text(
        row.累计票房, row.豆瓣评分, 
        row.电影名称, fontsize=15, 
        fontproperties=font
    ), axis=1
)

from sklearn.preprocessing import scale
#标准化数据
data['标准化累计票房'] = scale(data['累计票房'])
data['标准化豆瓣评分'] = scale(data['豆瓣评分'])

fig = plt.figure()

plt.xlabel('标准化累计票房', fontproperties=font)
plt.ylabel('标准化豆瓣评分', fontproperties=font)
plt.xlim([data['标准化累计票房'].min()*1.3, data['标准化累计票房'].max()*1.6])
plt.ylim([data['标准化豆瓣评分'].min()*1.3, data['标准化豆瓣评分'].max()*1.6])

plt.scatter(
    data['标准化累计票房'], 
    data['标准化豆瓣评分'], 
    alpha=0.5, s=200, marker="o", 
    facecolors='none', 
    edgecolors=mainColor, linewidths=5
)

data.apply(
    lambda row: plt.text(
        row.标准化累计票房, row.标准化豆瓣评分, 
        row.电影名称, fontsize=15, fontproperties=font
    ), axis=1
)

import numpy
plt.plot(
    numpy.arange(-5, 5), 
    numpy.arange(-5, 5),
    color=mainColor, linewidth=5
)

def verticalPoints(x, y):
    x0 = (x+y)/2
    y0 = (x+y)/2
    plt.scatter(
        x0, y0, 
        alpha=0.5, s=200, 
        marker="o", facecolors='none', 
        edgecolors=mainColor, linewidths=5
    )
    plt.plot(
        [x, x0], [y, y0],
        color=mainColor, linewidth=3
    )

data.apply(
    lambda row: verticalPoints(
        row.标准化累计票房, 
        row.标准化豆瓣评分
    ), axis=1
)
