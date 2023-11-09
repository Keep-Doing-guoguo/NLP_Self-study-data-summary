import os
import re
import csv
from collections import defaultdict
import numpy as np
from lstm import LSTM
#将词汇表加载为字典
def load_vocabulary(fpath = './runs/vocab'):#加载词汇表，这个词汇表应该是可以更换为大模型中的词汇表。
    vocabulary = {}
    with open(fpath, 'r',encoding='utf-8') as f:
        for line in f:
            split = line.strip('\n').split(' ')#去掉首位的空格，然后按照空格进行划分为列表。
            vocabulary[split[0]] = int(split[1])#加载为词典的格式。
    return vocabulary
vocabulary = load_vocabulary()#test
def context_win(words_index, wind_size):
    '''
    wind_size : int
      corresponding to the size of the window given a list of indexes composing a sentence对应窗口的大小给出一个索引列表，组成一个句子
    words_index : list
      array containing words index

    Return a list of indexes corresponding to contex windows surrounding each word
    in the sentence
    '''
    assert (wind_size % 2) == 1#用于检查一个条件是否为真。如果条件为真，程序将继续执行；如果条件为假，程序将抛出一个AssertionError异常并终止执行。
    assert wind_size >= 1
    words_index = list(words_index)

    lpadded = wind_size // 2 * [-1] + words_index + wind_size // 2 * [-1]#7//2 * 【-1】 + words_index + 7//2 * 【-1】.7//2取整
    out = [lpadded[i:(i+wind_size)] for i in range(len(words_index))]#制作训练集

    assert len(out) == len(words_index)
    return np.array(out, dtype=np.int32)

Status = ['B', 'M', 'E', 'S']

def load_data(fpath = './corpus/train.utf8', wind_size=7):#加载训练集信息

    X_train, Y_train = [], []
    vocabulary = load_vocabulary()#总共长度为5367.相当于token。
    with open(fpath,encoding='utf-8') as f:
        for line in f:
            split = re.split(r'\s+', line.strip())#首先去掉首位的空格，这是一个正则表达式。按照空格将line进行划分为列表。
            y = []
            for word in split:
                length = len(word)
                if length == 1:
                    y.append(3)#如果是单个字体的话，那么就设置3.
                else:
                    y.extend([0] + [1]*(length-2) + [2])#从0-1-1-1-2意味这0是开头，2是结尾。
            newline = ''.join(split)
            x = [vocabulary[char] if vocabulary.get(char) else 0 for char in newline]#转换为对应的数字。这里的y和上面的split是相同的。
            X_train.append(context_win(x, wind_size))
            Y_train.append(y)

    return X_train, Y_train, vocabulary

def train_with_sgd(model, X_train, y_train, learning_rate=0.001, nepoch=20, callback_every=10000, callback=None):

    num_example_seen = 0
    for epoch in range(nepoch):
        # For each training example...
        for i in np.random.permutation(len(y_train)):#生成一个随机排列的整数数组。
            # One SGD step
            if len(y_train[i]) < 3:#如果要是y的长度小于3，0，1，2，那就说明是一个字段。
                continue;
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_example_seen += 1
            if (callback and callback_every and num_example_seen % callback_every == 0):
                callback(model, num_example_seen)
    return model

def convert_predict_to_pos(predicts):
    pos_list = [Status[p] for p in predicts]
    return pos_list

def segment(predicts, sentence):
    pos_list = convert_predict_to_pos(predicts)
    assert len(pos_list) == len(sentence)
    words = []
    begin, nexti = 0, 0
    for i, char in enumerate(sentence):
        pos = pos_list[i]
        if pos == 'B':
            begin = i
        elif pos == 'E':
            words += [sentence[begin:i+1]]
            nexti = i + 1
        elif pos == 'S':
            words += [char]
            nexti = i + 1
    if nexti < len(sentence):
        words += [sentence[nexti:]]
    return words

def load_model(floder, modelClass, hyperparams):
    print("loading model from %s." % floder)
    print("...")

    E = np.load(os.path.join(floder, 'E.npy'))
    U = np.load(os.path.join(floder, 'U.npy'))
    W = np.load(os.path.join(floder, 'W.npy'))
    V = np.load(os.path.join(floder, 'V.npy'))
    b = np.load(os.path.join(floder, 'b.npy'))
    c = np.load(os.path.join(floder, 'c.npy'))

    hidden_dim = hyperparams['hidden_dim']
    embedding_dim = hyperparams['embedding_dim']
    vocab_size = hyperparams['vocab_size']
    num_clas = hyperparams['num_clas']
    wind_size = hyperparams['wind_size']

    model = modelClass(embedding_dim, hidden_dim, num_clas, wind_size, vocab_size)
    model.E.set_value(E)
    model.U.set_value(U)
    model.W.set_value(W)
    model.V.set_value(V)
    model.b.set_value(b)
    model.c.set_value(c)
    print("lstm model has been loaded.")
    return model
