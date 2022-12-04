# This is a second option to the one found in ch_4_02_partitioning_byclass 
# WE need to know how many samples in each subset
# This method simpler
# the possible downside is the mix of classes in each subset might not quite be the fractions we want.
import numpy as np
from sklearn.datasets import make_classification

x,y = make_classification(n_samples=10000, weights=(0.9,0.1))
idx = np.argsort(np.random.random(y.shape[0]))

# x values
x = x[idx]
# y values
y = y[idx]

# TRAINING
ntrn = int(0.9*y.shape[0])
# VALIDATION
nval = int(0.05*y.shape[0])

# TRAINING
xtrn = x[:ntrn]
ytrn = y[:ntrn]

# VALIDATION
xval = x[ntrn:(ntrn+nval)]
yval = y[ntrn:(ntrn+nval)]

# TEST
xtst = x[(ntrn+nval):]
ytst = y[(ntrn+nval):]