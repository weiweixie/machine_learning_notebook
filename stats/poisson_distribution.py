# -*- coding: utf-8 -*-
'''
Created on 2018/10/28 10:09
file : poisson_distribution.py

@author: xieweiwei
'''

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# poisson distribution
rate = 2
num = 10
n = np.arange(0, num)
y = stats.poisson.pmf(n, rate)

data = stats.poisson.rvs(mu=2, loc=0, size=10000)
print("Mean: %g" %np.mean(data))
print("SD: %g" %np.std(data,ddof=1))
plt.plot(n, y, 'o-')
plt.title('poisson: n=%i, rate=%.2f' % (num, rate), fontsize=15)
plt.xlabel('Number of success')
plt.ylabel('Probalility of success', fontsize=15)
plt.show()


plt.hist(data, bins=9, density=True)
plt.xlabel("Number of accidents")
plt.ylabel("Simulating Poisson Random Variables")
plt.show()
