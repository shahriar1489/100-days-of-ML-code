# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:34:53 2019

@author: Shahriar
"""

''' 
Sigmoid function: did many times before though
'''

import matplotlib.pyplot as plt 
import numpy as np

def sigmoid(z): 
    return 1.0 / (1.0 + np.exp(-z))

z = np.arange(-5, 5, 0.1)

phi_z = sigmoid(z)

plt.plot(z, phi_z)
plt.axvline(0.0, color='b') # have a look at the list of matplotlib colors 
'''
axline: vertical line at x = 0.0 in this case
'''
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi(z)$') # pay attention to the notation 
plt.yticks([0.0, 0.5,1.0])
'''
yticks -> marks at y = values
'''
# y axis ticks and gridline
ax = plt.gca()
ax.yaxis.grid(True)
ax.xaxis.grid(True) # this was 
plt.show()