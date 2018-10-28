# -*- coding: utf-8 -*-
'''
Created on 2018/10/28 10:09
file : norm_distribution.py

@author: xieweiwei
'''

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# norm distribution
mu = 0 #mean
sigma  =1  #standard sevaition
x = np.arange(-5,5,0.1)
y = stats.norm.pdf(x,0,1)


plt.plot(x, y)
plt.title('Normal: $\mu$=%.1f,$\sigma^2$=%.1f$' %(mu, sigma))
plt.xlabel('x')
# probability of observing each of these observations
plt.ylabel('Probalility density', fontsize=15)
plt.show()


