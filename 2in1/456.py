#!/usr/bin/env python  
# encoding: utf-8  
""" 
@author: GrH 
@contact: 1271013391@qq.com 
@file: 456.py 
@time: 2019/4/21 0021 12:45 
"""
import codecs
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import jieba # 结巴分词
# gensim用来加载预训练word vector
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
        self.file = codecs.open(path, 'r', encoding='utf-8')

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
    read = Read("small-test.txt")
    test_texts_orig = []
    id_list = []
    while True:
        id, string = read.getOne()
        if id is None:
            break
        id_list.append(id)
        test_texts_orig.append(string.strip())
    csvFile = open("C:/Users/Administrator/Desktop/001.csv", "w",newline='')  # 创建csv文件
    for i in range(len(id_list)):
        writer = csv.writer(csvFile)
        writer.writerow([id_list[i],1])
    csvFile.close()
