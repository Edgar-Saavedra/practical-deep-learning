# SCALING DATA
import numpy as np
from sklearn.datasets import make_classification
# Random Sampling
# Second common method to partition the full dataset via random sampling

# first randomize the full datase ant then extract
# the first 90% of samples as training
# %5 as validation set
# 5% test set
# x = samples
# y = labels
x, y = make_classification(n_samples=10000, weights=(0.9,0.1))
idx = np.argsort(np.random.rand(y.shape[0]))
x = x[idx]
y = y[idx]

# training set is 90%
ntrn = int(0.9*y.shape[0])
# validation is 5%
nval = int(0.05*y.shape[0])

# training data
# print(x[:ntrn].shape)
xtrn = x[:ntrn] # sampels
ytrn = y[:ntrn] # labels

# validation data
# print(x[ntrn:(ntrn+nval)].shape)
xval = x[ntrn:(ntrn+nval)] # sampels
yval = y[ntrn:(ntrn+nval)] # labels

# test data
# print(x[(ntrn+nval):].shape)
xtst = x[(ntrn+nval):] # sampels
ytst = y[(ntrn+nval):] # labels

# Although this is simpler than 4-1.py it can cause the fractions of classes to be off
# sometimes this is not a problems other times this can be problamatic