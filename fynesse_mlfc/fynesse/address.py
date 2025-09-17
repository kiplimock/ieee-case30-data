# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""


import numpy as np
import torch
import torch.nn as nn

def train_lm(inp, out, model, mode='lasso', alpha=1e-5):
    """
    Trains linear models (e.g., Lasso, Ridge, Linear Regression) for each node in the power grid.
    Args:  
        inp (np.ndarray): Input features of shape (samples, nodes, features).
        out (np.ndarray): Output targets of shape (samples, nodes, 2) where the last dimension represents (magnitude, angle).
        model (class): A linear model class from sklearn (e.g., Lasso, Ridge, LinearRegression).
        mode (str): Type of model to train ('lasso', 'ridge', 'linear').
        alpha (float): Regularization strength for Lasso or Ridge.
    Returns:
        (list, list): Two lists containing trained models for magnitudes and angles for each node.
    """

    node_count = out.shape[1]
    mag_models, ang_models = [], []

    # find index of nodes with nonzero loads and use as input to model
    nonzero_indx = np.abs(inp[:,:,:2].mean(0)).sum(1) != 0 # use only two features (P, Q)
    inp_reshaped = inp[:,nonzero_indx,:2].reshape(-1, nonzero_indx.sum()*2) # --> (samples, n_features*n_nonzero_buses)

    # out[:,i,0] --> [n_samples, ]
    # For all nonzero [P, Q] at all nodes in the grid, there is a model relationship to the [Vm, Va] at
    # some target node

    for i in range(node_count): # results in n_buses models for mag and ang
        if mode == 'lasso':
            mag_models.append(model(alpha=alpha).fit(inp_reshaped, out[:,i,0]))
            ang_models.append(model(alpha=alpha).fit(inp_reshaped, out[:,i,1]))
        else:
            mag_models.append(model().fit(inp_reshaped, out[:,i,0]))
            ang_models.append(model().fit(inp_reshaped, out[:,i,1]))
    
    return (mag_models, ang_models)


def get_predictions(models, inp):
    shape = inp.shape[:2]
    ym, ya = np.zeros(shape), np.zeros(shape)
    nonzero_indx = np.abs(inp[:,:,:2].mean(0)).sum(1) != 0
    inp_reshaped = inp[:,nonzero_indx,:2].reshape(-1, nonzero_indx.sum()*2) # --> (samples, n_features*n_nonzero_buses)
    
    print(inp_reshaped.shape, inp.shape)
    
    # We use the [P, Q] at all nodes in the grid to predict [Vm, Va] at some target node
    # The same input tensor of shape [samples, n_features*n_nonzero_buses] is used to predict 
    # an output tensor (Vm or Va) of shape [n_samples,]
    for i in range(shape[1]): # prediction for each bus yields array of shape [n_samples,]
        ym[:,i] = models[0][i].predict(inp_reshaped)
        ya[:,i] = models[1][i].predict(inp_reshaped)
    
    return (ym, ya)


# Simple MLP model for comparison
class MLP(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers=3, dp_rate=0.1):
        super().__init__()
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for _ in range(num_layers-2):
            layers += [
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [nn.Linear(in_channels, c_out),
                   nn.ReLU(inplace=True),
                   nn.Linear(c_out, c_out)
                   ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)