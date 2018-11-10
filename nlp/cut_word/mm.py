# -*- coding: utf-8 -*-
'''
Created on 2018/11/8 22:43
file : Imm.py

@author: xieweiwei
'''


class MM:
    def __init__(self, dic_path):
        self.dictionary = set()
        self.maximum = 0
        with open(dic_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if line is None:
                    continue
                self.dictionary.add(line)
                self.maximum = max(self.maximum, len(line))

    def cut(self, text):
        result = []
        index = 0
        length = len(text)

        while index < length - 1:
            # 每次分词前需要设置一个flag，如果本轮分词失败就跳过一个索引
            word = None
            for size in range(self.maximum, 0, -1):
                if index + size > length:
                    continue
                piece = text[index: index+size]
                if piece in self.dictionary:
                    word = piece
                    result.append(word)
                    index += size
                    break
            if word is None:
                result.append(text[index])
                index += 1

        return result


if __name__ == "__main__":
    text = "南京市长江大桥"
    dic_path = r'./data/trainCorpus.txt_utf8'
    # dic_path = r'./data/imm_dic.utf8'
    tokenizer = MM(dic_path)
    result = tokenizer.cut(text)
    print(result)
