# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 17:09:56 2019

@author: Shahriar 

"""

'''
Agenda: Module and Class
'''

class Square: 
    side = 0
    def area(self): # unless self was passed, error was returned
        return self.side * self.side
    
    
ob = Square()

print(ob.area())



class Sq: 
    side = 0 
    def __init__(self, x):
        print('print constructor function')  # prints this line
        self.side = x 
        print('side = ', self.side)
        print('self.side updated successfully\n\n')
    def area(self): 
        return self.side * self.side
    
ob = Sq(4)

print(ob.area())    # works