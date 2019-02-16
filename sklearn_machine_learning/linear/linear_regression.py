# -*- coding: utf-8 -*-
'''
Created on 2019/2/16 22:27
file : linear_regression.py

@author: xieweiwei
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn import  datasets,linear_model
from sklearn.model_selection import  train_test_split

def load_data():
    """
      load the dataset for problem of regression
    :return: 一个元组，元组元素依次为：训练样本集、测试样本集、训练样本集对应的值、测试样本集对应的值
    """
    diabetes = datasets.load_diabetes()
    return train_test_split(diabetes.data,diabetes.target,
                            test_size=0.25,random_state=0)

def linear_regression_test(*data):
    """

    :param data:
    :return:
    """
    x_train,x_test,y_train,y_test=data
    regr = linear_model.LinearRegression()
    regr.fit(x_train,y_train)
    print('Coefficients:%s, intercept %.2f' % (regr.coef_, regr.intercept_))
    print("Residual sum of squares: %.2f" % np.mean((regr.predict(x_test) - y_test) ** 2))
    print('Score: %.2f' % regr.score(x_test, y_test))

if __name__ == "__main__":
    x_train,x_test,y_train,y_test=load_data()
    linear_regression_test(x_train,x_test,y_train,y_test)