# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 19:52:10 2019

@author: Shariar
"""
import numpy as np

nx, ny = (3, 2)
print('nx = ', nx )
print('ny = ', ny )

print('type(nx) = ', type(nx) )
print('type(ny) = ', type(ny) )


x = np.linspace(0, 1, nx) # nx = 3 
y = np.linspace(0, 1, ny) # ny = 2

xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
print('xv = \n', xv);
print('yv = \n', yv, '\n');

xv, yv = np.meshgrid(x, y, sparse=True)
print(xv, '\n'); 
print(yv, '\n');

''' 
Q. sparse? 
Ans. Some kind of dimensionality reduction. Need to look at this.
'''

import matplotlib.pyplot as plt
x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)

xx, yy = np.meshgrid(x, y, sparse=True)
z = np.sin(xx**2 + yy**2)
#h = 
plt.contourf(x ,y, z)  
'''
Note: not allocating the the function above to 'h' does not make difference in 
output 
'''
#print('type(h) = ', type(h))
#print('h = ', h)
plt.show()