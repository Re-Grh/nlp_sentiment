#!/usr/bin/env python  
# encoding: utf-8  
""" 
@author: GrH 
@contact: 1271013391@qq.com 
@file: 123.py 
@time: 2019/4/20 0020 23:01 
"""
import codecs
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import jieba
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, LSTM, Bidirectional
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore")
import csv
class Read:
    def __init__(self, path):
        self.file = codecs.open(path, 'r',encoding='utf-8')


    def getOne(self):
        string = self.file.readline()
        id = 0
        while True:
            if not string:
                return None, None
            id_strings = re.findall("\d+", string)
            if len(id_strings) == 1:
                id = int(id_strings[0])
                break
            string = self.file.readline()
        result = ""
        while True:
            string = self.file.readline()
            if str.startswith(string, "</review>"):
                break
            result += string[:-1]
        return id, result


if __name__ == "__main__":
    read1 = Read("sample.positive.txt")
    read2 = Read("sample.negative.txt")

    train_texts_orig = []
    while True:
        id, string = read1.getOne()
        if id is None:
            break
        train_texts_orig.append(string.strip())
    while True:
        id, string = read2.getOne()
        if id is None:
            break
        train_texts_orig.append(string.strip())

    # 使用gensim加载预训练中文分词embedding
    cn_model = KeyedVectors.load_word2vec_format('sgns.zhihu.bigram',
                                                 binary=False)
    embedding_dim = 300
    # 进行分词和tokenize
    # train_tokens是一个长长的list，其中含有10000个小list，对应每一条评价
    train_tokens = []
    for text in train_texts_orig:
        # 去掉标点
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
        # 结巴分词
        cut = jieba.cut(text)
        # 结巴分词的输出结果为一个生成器
        # 把生成器转换为list
        cut_list = [i for i in cut]
        for i, word in enumerate(cut_list):
            try:
                # 将词转换为索引index
                cut_list[i] = cn_model.vocab[word].index
            except KeyError:
                # 如果词不在字典中，则输出0
                cut_list[i] = 0
        train_tokens.append(cut_list)

    # 获得所有tokens的长度
    num_tokens = [len(tokens) for tokens in train_tokens]
    num_tokens = np.array(num_tokens)
    # 取tokens平均值并加上两个tokens的标准差，
    # 假设tokens长度的分布为正态分布，则max_tokens这个值可以涵盖97%左右的样本
    max_tokens = np.mean(num_tokens) + 3 * np.std(num_tokens)
    max_tokens = int(max_tokens)
    # print(np.sum(num_tokens < max_tokens) / len(num_tokens))

    num_words = 80000
    # 初始化embedding_matrix，之后在keras上进行应用
    embedding_matrix = np.zeros((num_words, embedding_dim))
    # embedding_matrix为一个 [num_words，embedding_dim] 的矩阵
    # 维度为 80000 * 300
    for i in range(num_words):
        embedding_matrix[i, :] = cn_model[cn_model.index2word[i]]
    embedding_matrix = embedding_matrix.astype('float32')
    # 进行padding和truncating， 输入的train_tokens是一个list
    # 返回的train_pad是一个numpy array
    train_pad = pad_sequences(train_tokens, maxlen=max_tokens,
                              padding='pre', truncating='pre')
    # 超出词向量的词用0代替
    train_pad[train_pad >= num_words] = 0
    # 准备target向量，前5000样本为1，后5000为0
    train_target = np.concatenate((np.ones(5000), np.zeros(5000)))
    # 90%的样本用来训练，剩余10%用来测试
    X_train, X_test, y_train, y_test = train_test_split(train_pad,
                                                        train_target,
                                                        test_size=0.1,
                                                        random_state=0)
    # 用LSTM对样本进行分类
    model = Sequential()
    # 模型第一层为embedding
    model.add(Embedding(num_words,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=max_tokens,
                        trainable=True))

    model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
    model.add(LSTM(units=16, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    # # 我们使用adam以0.001的learning rate进行优化
    # optimizer = Adam(lr=1e-3)
    # model.compile(loss='binary_crossentropy',
    #               optimizer=optimizer,
    #               metrics=['accuracy'])
    #
    # # 模型结构
    # model.summary()
    #
    # # 建立一个权重的存储点
    # path_checkpoint = 'sentiment_checkpoint.keras'
    # checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss',
    #                              verbose=1, save_weights_only=True,
    #                              save_best_only=True)
    #
    # # 定义early stoping如果3个epoch内validation loss没有改善则停止训练
    # earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    # # 自动降低learning rate
    # lr_reduction = ReduceLROnPlateau(monitor='val_loss',
    #                                  factor=0.1, min_lr=1e-5, patience=0,
    #                                  verbose=1)
    # # 定义callback函数
    # callbacks = [
    #     earlystopping,
    #     checkpoint,
    #     lr_reduction
    # ]
    # # 开始训练
    # model.fit(X_train, y_train,
    #           validation_split=0.1,
    #           epochs=20,
    #           batch_size=128,
    #           callbacks=callbacks)
    # result = model.evaluate(X_test, y_test)
    # print('Accuracy:{0:.2%}'.format(result[1]))

    read = Read("test.txt")
    test_texts_orig = []
    id_list=[]
    predict_list = []
    while True:
        id, string = read.getOne()
        if id is None:
            break
        id_list.append(id)
        test_texts_orig.append(string.strip())


    def predict_sentiment(text):
        # print(text)
        # 去标点
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
        # 分词
        cut = jieba.cut(text)
        cut_list = [i for i in cut]
        # tokenize
        for i, word in enumerate(cut_list):
            try:
                cut_list[i] = cn_model.vocab[word].index
            except KeyError:
                cut_list[i] = 0
        # padding
        tokens_pad = pad_sequences([cut_list], maxlen=max_tokens,
                                   padding='pre', truncating='pre')
        model.load_weights('sentiment_checkpoint.keras')
        # 预测
        result = model.predict(x=tokens_pad)
        coef = result[0][0]
        if coef >= 0.5:
            return 1
        else:
            return 0
    for i in range(len(id_list)):
        predict_list.append(predict_sentiment(test_texts_orig[i]))
    csvFile = open("C:/Users/Administrator/Desktop/1150310123.csv", "w", newline='')  # 创建csv文件
    for i in range(len(id_list)):
        writer = csv.writer(csvFile)
        writer.writerow([id_list[i],predict_list[i]])
    csvFile.close()