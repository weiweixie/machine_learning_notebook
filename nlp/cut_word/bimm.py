# -*- coding: utf-8 -*-
'''
Created on 2018/11/10 7:53
file : bimm.py

@author: xieweiwei
'''
from nlp.cut_word.Imm import IMM
from nlp.cut_word.mm import MM


class BiMM:
    def __init__(self, dic_path):
        self.mm = MM(dic_path)
        self.imm = IMM(dic_path)

    def cut(self, text):
        mm_result = self.mm.cut(text)
        imm_result = self.imm.cut(text)
        if len(mm_result) < len(imm_result):
            return mm_result
        else:
            return imm_result

if __name__ == "__main__":
    text = "南京市长江大桥"
    # dic_path = r'./data/trainCorpus.txt_utf8'
    dic_path = r'./data/imm_dic.utf8'
    tokenizer = BiMM(dic_path)
    result = tokenizer.cut(text)
    print(result)