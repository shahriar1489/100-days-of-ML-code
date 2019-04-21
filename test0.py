# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:09:15 2019

@author: Shahriar
"""

import numpy as np 
import matplotlib.pyplot as plt

a = np.arange(0, 11)
print('a: ', a)

''' 
Agenda
1. Plot a linear function: DONE
2. Compute x and y coordinates for points on sine and cosine curves: DONE
''' 

x_ = np.arange(0, 1, 0.1)
#y_ = np.empty #create empty numpy array
#y_ = np.arange(0, 1, 0.1)
y_  = 3*x_ - 2 # declaring the linear equation 
y1_ = x_ - 2

#for i in x_: 
#    y.add(i)
    
plt.plot(x_, y_)
plt.plot(x_, y1_)
plt.xlabel('x axis label')
plt.ylabel('y axis label')

plt.legend(['y = 3x - 2', 'y = x - 2'])

plt.title('Linear Eqn Plot')

plt.show()

x = np.arange(0, 3 * np.pi, 0.1)

y_sin = np.sin(x)
y_cos = np.cos(x)

# plot the coordinates using matplotlib 
plt.plot(x, y_sin)
plt.plot(x, y_cos)

plt.xlabel('x axis label')
plt.ylabel('y axis label')

plt.legend(['Sine', 'Cosine'])

plt.show()