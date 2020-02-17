""" _________________________ Importing Module(s) __________________________"""


import numpy as np
import matplotlib.pyplot as plt


""" ____________________ Artificial Neural Network Setup ______________________
                                                                              .
Input layer        | Hidden layer | Output layer (P[Target=1])|               .
     {}                                                                       .
     {}                 {}                                                    .
                                                           |`````````|        .
                                                    {} --> |Threshold| >: Yes .
     .                  .                                  |_________| <: No  .
     .                  .                                                     .
     .                  .                                                     .
     {}                 {}                                                    .
     {}                 {}                                                    .
D + 1(bias) Neurons | h + 1(bias) Neurons in the hidden layer | 1 Neuron      .
____________________________________________________________________________"""


class Network:
    def __init__(self, dim, activ):
        '''
        Args:
            dim (any iterable): Dimensions of the neural net where the element
            at index i of the iterable denotes the number of nodes in the
            corresponding layer: (num_features+1, h+1,n_o);
            activ (any iterable) : Activation function(s) to be applied to
            neurons in the order of layers; e.g., ReLU for the hidden layer and
            Sigmoid for the output layer.

        Example architecture:
        - Input layer: 4+1 Neurons
        - One hidden layer: 7+1 Neurons
        - Output layer: 1 Neuron
        Layer               |  1      2         3
        ------------------------------------------
        # of neurons (dim)  | [5,     8,        1]
        Activation (active) |   (Sigmoid/ReLU, Sigmoid)
        '''

        self.num_layers = len(dim)
        self.loss = None
        self.learning_rate = None

        '''   _ Setup & Initialization of Weights,Biases, & Activations _   '''
        #  Keys are the layers with 1 being input, 2 hidden, and 3, the output.
        self.W = {}  # W={1:W[1] 2:W[2]}

        self.activ = {}  # active = {2:'Sigmoid/ReLu', 3:'Sigmoid'}

        for l in range(len(dim) - 1):  # l= 0, 1
            self.W[l + 1] = np.random.uniform(-0.1, 0.1, (dim[l+1], dim[l]))
            self.activ[l + 2] = activ[l]  # no activation applied to inputs

    '''_____________________________________________________________________'''
    def _feed_forward(self, x):
        '''
        Forward propagation.
        Args:
            x: Augmented input feature vector.
        Return:
            z(dict): Input or output of the activation functions
            a(dict): Weighted sum of inputs to a neuron AKA activation
            y(float): Predicted output, probability.
        '''
        # First layer has no activationfunction and x is the augmented input:
        z = {1: x}
        a = {}

        # Hidden layer weighted sums (activations) and z's:
        a[2] = np.dot(self.W[1], z[1])
        z[2] = self.activ[2].activation(a[2])
        z[2][0] = 1  # bias neuron

        # Output layer weighted sum (activation) and z (predicted output):
        a[3] = np.dot(self.W[2], z[2])
        z[3] = self.activ[3].activation(a[3])
        return z, a

    '''_____________________________________________________________________'''
    def _back_prop(self, z, a, t):
        '''
        Args:
        z = { 1: x,
              2: Sigmoid/ReLu(W[1]x)
              3: Sigmoid(W[2]a[2]) | Predicted output
              }
        a = { 2: W[1]x
              3: W[2]z[2]
              }

        t(int {0,1}) True label.
        '''

        # Determine delta and partial derivative for the output layer:
        delta = z[self.num_layers]-t
        dw = delta * z[self.num_layers - 1].T

        updates = {2: dw}

        '''Backpropagate the delta of output layer to obtain delta for each
           neuron in the hidden layer and determine the partial derivative.'''
        delta = np.multiply(np.dot(self.W[2].T, delta),
                            self.activ[2].deriv(a[2]))
        dw = np.outer(delta, z[1])
        updates[1] = dw

    # Update the weights
        for lay, dw in updates.items():
            self._update_w_b(lay, dw)

    def _update_w_b(self, layer, dw):
        '''
        Update weights and biases according to stochastic gradient descent.
        Args:
            layer (int): Number of the layer
            dw (array): Partial derivatives of ce w.r.t. the weights
        '''

        self.W[layer] -= self.learning_rate * dw

    '''_____________________________________________________________________'''

    def train(self, x_vec, t_vec, loss, epochs, learning_rate):
        """
        Train the neural network.
        Args:
        x_vec (pandas DataFrame): Augmented feature vectors.
        t_vec (pandas DataFrame): Containing biary labels.
        loss: Loss class (CrossEntropy in case of classification)
        epochs (int): Number of epochs for SGD.
        learning_rate (float)
        """
        if not x_vec.shape[0] == t_vec.shape[0]:
            raise ValueError("Dimension mismatch!")

        # Initiate the loss object with the final activation function
        self.loss = loss(self.activ[self.num_layers])
        self.learning_rate = learning_rate
        ce_loss = []
        for iter_ in range(epochs):
            dummy_ce = 0
            # Shuffle the data
            shuffle = np.random.permutation(len(x_vec))
            x_ = x_vec.iloc[shuffle]
            t_ = t_vec.iloc[shuffle]

            for ex in range(x_vec.shape[0]):
                example = x_.iloc[ex, :]
                z, a = self._feed_forward(example)
                dummy_ce += self.loss.loss(t_.iloc[ex], z[self.num_layers])
                self._back_prop(z, a, t_.iloc[ex])
            dummy_ce /= x_vec.shape[0]
            #  OPT: print("Iteration Count: ",iter_)
            ce_loss.append(dummy_ce)
            '''___________________ Log & Plot CE Loss ______________________'''
            # OPT:
            '''
            if (iter_ + 1) % 100 == 0:
                print(iter_+1)
                print("Loss:", dummy_ce)
            '''
        plt.figure(1)
        plt.plot(range(1, epochs+1), ce_loss)
        plt.yscale('log')
        plt.xlim([1.0, epochs])
        plt.rcParams['font.size'] = 12
        plt.title('Normalized Cross-entropy vs. Epochs Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Cross-entropy Averaged over Training Examples')
        plt.grid(True)

    '''_____________________________________________________________________'''
    def test(self, x_vec, t_vec, threshold):
        """
        Args:
            x_vec (pandas DataFrame): Augmented feature vectors
            threshold (float between 0 and 1)
        Return:
            y_pred (list) A n_test-dim list of {0,1}.
        """
        y_pred = []
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for ex in range(x_vec.shape[0]):
            z, a = self._feed_forward(x_vec.iloc[ex, :])
            y_pred.append(int((z[self.num_layers] > threshold).item(0)))
            if t_vec.iloc[ex] == 1 and y_pred[ex] == 1:
                tp += 1
            elif t_vec.iloc[ex] == 0 and y_pred[ex] == 1:
                fp += 1
            elif t_vec.iloc[ex] == 0 and y_pred[ex] == 0:
                tn += 1
            else:
                fn += 1
        return y_pred, tp, tn, fp, fn
