# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 16:27:13 2019

@author: Shahriar
"""

import numpy as np
import matplotlib.pyplot as plt

# Compute sine and cosine
x = np.arange(0, 3* np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)
y_tan = np.tan(x)
#Set up a subplot grid w/ height 2 and width 1, 
# and set the first such subplot as active
plt.subplot(2, 1, 1) 
#Make first plot
plt.plot(x, y_sin)
plt.title('Sine')

plt.show()
#Set the second subplot
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

plt.show()

#Set THIRD subplot 
plt.subplot(2, 1, 2)

plt.plot(x, y_tan)
plt.title('Tangent')

plt.show()

'''
See subplot documentation for parameters
'''