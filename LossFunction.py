""" _________________________ Importing Module(s) __________________________"""


import numpy as np


"""_______________________ Loss Function: Cross Entropy ____________________"""


class CE:
    def __init__(self, activ_fn):
        '''
        Args:
            activ_fn: Object of class activation function; e.g., Sigmoid or
            ReLU
        '''
        self.activ_fn = activ_fn

    def activation(self, a):
        return self.activ_fn.activation(a)

    @staticmethod
    def loss(t, y):
        '''
        Args:
            t (int: {0,1}) True label.
            y (float: [0,1]):  Generated output, probability!
        Return:
            ce (float): Cross entropy.
        '''
        if t == 0:
            ce = -(1-t)*np.log(1-y+1e-15)
        else:
            ce = -t*np.log(y+1e-15)
        return ce


"""_________________________________________________________________________"""
