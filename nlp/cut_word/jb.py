# -*- coding: utf-8 -*-
'''
Created on 2018/11/8 22:43
file : Imm.py

@author: xieweiwei
'''
import glob
import random

import jieba


class JB:
    def __init__(self):
        pass

    @classmethod
    def get_content(cls, path):
        """
        返回路径下的内容
        :param self:
        :param path:
        :return: 字符串
        """
        with open(path, 'r', encoding='gbk', errors='ignore') as f:
            content = ''
            for line in f:
                line = line.strip()
                content += line
            return content

    @classmethod
    def get_TF(cls, words, topK=10):
        tf_dic = {}
        for w in words:
            tf_dic[w] = tf_dic.get(w, 0) + 1
        return sorted(tf_dic.items(), key=lambda x: x[1], reverse=True)[:topK]

    @classmethod
    def stop_words(cls, path):
        with open(path, encoding='utf8') as f:
            return [line.strip() for line in f]


if __name__ == "__main__":
    files = glob.glob(r'./data/news/C000013/*.txt')
    stop_words_path = r'./data/stop_words.utf8'
    corpus = [JB.get_content(x) for x in files]

    sample_inx = random.randint(0, len(corpus))
    split_words = [x for x in jieba.cut(corpus[sample_inx])
                   if x not in JB.stop_words(stop_words_path)]
    print("Sample 1: ", corpus[sample_inx])
    print("\nAfter Cutting: ", '/'.join(split_words))
    print("\nTopK(10) words: ", str(JB.get_TF(split_words)))

