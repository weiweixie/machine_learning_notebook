# -*- coding: utf-8 -*-
'''
Created on 2018/11/8 22:43
file : Imm.py

@author: xieweiwei
'''


class IMM:
    def __init__(self, dic_path):
        self.dictionary = set()
        self.maximum = 0

        # 读取词典
        with open(dic_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    # 去除空白行
                    continue
                self.dictionary.add(line)
                self.maximum = max(self.maximum, len(line))

    def cut(self, text):
        result = []  # 分词结果
        index = len(text)  # 待分词的字符串长度
        while index > 0:  # 循环指导整个字符串分词完毕
            word = None
            # 在待分词字符串中找出一个最大的长度进行尝试
            for size in range(self.maximum, 0, -1):
                if index - size < 0:
                    continue
                # 按当前字符串的最大可行长度尝试划分
                piece = text[index - size: index]
                # 如果划分出来的部分在词典中，那么本轮尝试结束，跳到下一轮
                # 因为要重新根据新索引找合适的size，所以需要break这个for循环
                if piece in self.dictionary:
                    word = piece
                    result.append(word)
                    # 更新下一轮的末尾索引
                    index -= size
                    break
            # 如果在该index下分词都失败，那就只能跳过这个index，往前挪一步
            if word is None:
                index -= 1
                result.append(text[index])

        # 如果index <= 0，说明只剩一个或者不剩未分词的字符串，分词结束
        return result[::-1]  # 双冒号表示步长，负数表示反向


if __name__ == "__main__":
    text = "南京市长江大桥"
    # dic_path = r'./data/trainCorpus.txt_utf8'
    dic_path = r'./data/imm_dic.utf8'
    tokenizer = IMM(dic_path)
    result = tokenizer.cut(text)
    print(result)
