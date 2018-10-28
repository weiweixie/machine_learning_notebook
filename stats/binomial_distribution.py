# -*- coding: utf-8 -*-
'''
Created on 2018/10/28 10:09
file : binomial_distribution.py

@author: xieweiwei
'''

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# binomial distribution

n = 10
p = 0.3
k = np.arange(0, 21)
binomial = stats.binom.pmf(k, n, p)

print(binomial)

plt.plot(k, binomial, 'o-')
plt.title('Binomial: n=%i, p=%.2f' % (n, p), fontsize=15)
plt.xlabel('Number of success')
plt.ylabel('Probalility of success', fontsize=15)
plt.show()

binom_sim = data = stats.binom.rvs(n=10, p=0.3, size=10000)
print("Mean: %g" % np.mean(binom_sim))
print("SD: %g" % np.std(binom_sim, ddof=1))
plt.hist(binom_sim, bins=10, density=True)
plt.xlabel("x")
plt.ylabel("density")
plt.show()
