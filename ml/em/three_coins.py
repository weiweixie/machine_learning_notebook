# -*- coding: utf-8 -*-
'''
Created on 2018/10/5 14:33
file : three_coins.py

@author: xieweiwei
'''

import math


def cal_u(pi, p, q, xi):
    """
      u值计算
    :param pi: 下一次迭代开始的 pi
    :param p:  下一次迭代开始的 p
    :param q:  下一次迭代开始的 q
    :param xi: 观察数据第i个值，从0开始
    :return:
    """
    return pi * math.pow(p, xi) * math.pow(1 - p, 1 - xi) / \
           float(pi * math.pow(p, xi) * math.pow(1 - p, 1 - xi) +
                 (1 - pi) * math.pow(q, xi) * math.pow(1 - q, 1 - xi))

def e_step(pi,p,q,x):
    """
        e步计算
    :param pi: 下一次迭代开始的 pi
    :param p:  下一次迭代开始的 p
    :param q:  下一次迭代开始的 q
    :param x: 观察数据
    :return:
    """
    return [cal_u(pi,p,q,xi) for xi in x]

def m_step(u,x):
    """
     m步计算
    :param u:  m步计算的u
    :param x:  观察数据
    :return:
    """
    pi1=sum(u)/len(u)
    p1=sum([u[i]*x[i] for i in range(len(u))]) / sum(u)
    q1=sum([(1-u[i])*x[i] for i in range(len(u))]) / sum([1-u[i] for i in range(len(u))])
    return [pi1,p1,q1]

def run(observed_x, start_pi, start_p, start_q, iter_num):
    """

    :param observed_x:  观察数据
    :param start_pi:  下一次迭代开始的pi $\pi$
    :param start_p:  下一次迭代开始的p
    :param start_q:  下一次迭代开始的q
    :param iter_num:  迭代次数
    :return:
    """
    for i in range(iter_num):
        u=e_step(start_pi, start_p, start_q, observed_x)
        print (i,[start_pi,start_p,start_q])
        if [start_pi,start_p,start_q]==m_step(u, observed_x):
            break
        else:
            [start_pi,start_p,start_q]=m_step(u, observed_x)

if __name__ =="__main__":
    # 观察数据
    x = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
    # 初始化 pi，p q
    [pi, p, q] = [0.4, 0.6, 0.7]
    # 迭代计算
    run(x,pi,p,q,100)