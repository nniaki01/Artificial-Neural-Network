""" _________________________ Importing Module(s) __________________________"""


import numpy as np


"""________________________ ReLU Activation Function _______________________"""


class ReLU:
    @staticmethod
    def activation(a):
        a[a < 0] = 0
        return a

    @staticmethod
    def deriv(a):
        a[a > 0] = 1
        a[a <= 0] = 0
        return a


"""_________________________________________________________________________"""

"""______________________ Sigmoid Activation Function ______________________"""


class Sigmoid:
    @staticmethod
    def activation(a):
        a[a < -50] = -50  # to avoid numerical issues
        return 1 / (1 + np.exp(-a))

    @staticmethod
    def deriv(a):
        return Sigmoid.activation(a) * (1 - Sigmoid.activation(a))


"""_________________________________________________________________________"""
