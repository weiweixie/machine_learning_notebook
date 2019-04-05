# -*- coding: utf-8 -*-
'''
Created on 2019/1/29 22:48
file : knn_kd.py

@author: xieweiwei
'''

T = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]


class node:
    def __init__(self, point):
        self.left = None
        self.right = None
        self.point = point
        pass


def median(lst):
    m = int (len(lst) / 2)
    return lst[m], m


def build_kdtree(data, d):
    data = sorted(data, key=lambda x: x[d])
    p, m = median(data)
    tree = node(p)

    del data[m]
    print(data, p)

    if m > 0: tree.left = build_kdtree(data[:m], not d)
    if len(data) > 1: tree.right = build_kdtree(data[m:], not d)
    return tree


kd_tree = build_kdtree(T, 0)
print(kd_tree)