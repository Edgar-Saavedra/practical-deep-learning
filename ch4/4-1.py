# If there are 9,000 samples from class 0
# 1000 samples for class 1. 
# We want to put 90%  of data into training 5% each validation/test at random
# Training set: Select 8,100 from class 0 at random
# Training set: select 900 at random from class 1

# Validation set : Select 450 at random from class 0
# Validation set : Select 50 at random from class 1

# Test set : Select 450 at random from class 0
# Test set : Select 50 at random from class 1

# 90/5/5 split
import numpy as np
from sklearn.datasets import make_classification

a,b = make_classification(n_samples=10000, weights=(0.9,0.1))

# https://numpy.org/doc/stable/reference/arrays.indexing.html
# split into class 0 and class 1 collections
# class 0
idx = np.where(b == 0)[0]
# print(idx);
x0 = a[idx, :]
y0 = b[idx]
# print(idx.shape, x0.shape, y0.shape)

# class 1
idx = np.where(b == 1)[0]
# print(idx.shape)
# get all rows and columns
x1 = a[idx, :]
y1 = b[idx]

