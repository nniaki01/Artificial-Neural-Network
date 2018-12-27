""" __________________________ Importing Modules ___________________________"""


import numpy as np
import preprocess_data as pp
import ActivationFunctions as af
import LossFunction as lf
import ANN as ann
from EvaluationCurves import confusion_matrix


"""_________________________________________________________________________"""

""" ___________________________________________________________________________
Code sample to be submitted to Wayfair.                                       .
Sent on Tue Dec.18.2018.                                                      .
@author: Fakhteh Saadatniaki                                                  .
                                                                              .
Project: Aritificial Neural Network (ANN) for binary classification.          .
Datasets: Banknote Authentication | Pima Indians Diabetes Database            .
____________________________________________________________________________"""
np.random.seed(1)
"""_________________________________________________________________________"""


if __name__ == '__main__':
    ds_id = int(input('Enter 0 for Banknote and 1 for Diabetes: '))
    if not ds_id:
        ''' Banknote '''
        features_train, labels_train, features_test, labels_test,\
            yes_examples, no_examples = pp.banknote('banknote.csv')
        h = 7
    else:
        ''' Diabetes '''
        features_train, labels_train, features_test, labels_test,\
            yes_examples, no_examples = pp.diabetes('diabetes.csv')
        h = 15

    # OPT: h = int(input('Enter the number of the hidden layer neurons: '))
    # OPT: epochs = int(input('Enter the number of SGD epochs: '))
    # OPT: learning_rate = int(input('Enter the SGD learning rate: '))
    nn = ann.Network(
            (features_train.shape[1], h+1, 1), (af.Sigmoid, af.Sigmoid))
    nn.train(features_train, labels_train, loss=lf.CE, epochs=5,
             learning_rate=1e-2)
    status = confusion_matrix(yes_examples, no_examples, nn, features_test,
                              labels_test, ds_id)
