# https://github.com/rkneusel9/PracticalDeepLearningPython
# we want to determin the number of samples representing each class and set aside selected percentages of each
# We want to put 90 percent of the data into traingin and 5% each into validation and tes
# at random for the traing set
# at random for the validation
import numpy as np
from sklearn.datasets import make_classification

# create dummy dataset
a, b = make_classification(n_samples=10000, weights=(0.9, 0.1))

# split into class 0
idx = np.where(b == 0)[0]

# samples (vectors)
# take all x samples
x0 = a[idx, :]
# labels, get all x labels
y0 = b[idx]
# print('idx', idx)
# print('x0', x0)
# print('y0', y0)

# split into class 1
idx = np.where(b == 1)[0]
# samples (vectors)
# take all y samples
x1 = a[idx, :]
# labels, get all y labels
y1 = b[idx]
# print('idx', idx)
# print('x1', x1)
# print('y1', y1)

# Randomize ordering, pull the first n samples withouth worrying we might be introducing bias
idx = np.argsort(np.random.random(y0.shape))
# vectors
x0 = x0[idx]
# labels
y0 = y0[idx]

idx = np.argsort(np.random.random(y1.shape))
# vectors
x1 = x1[idx]
# labels
y1 = y1[idx]


# --- TRAINING SET 90% of the samples for the the two classes
# class 0
ntrn0 = int(0.9*x0.shape[0])
# class 1
ntrn1 = int(0.9*x1.shape[0])
# 8062 937
# print(ntrn0, ntrn1)

# Return a new array of given shape and type, filled with zeros.
# numpy.zeros - Return a new array of given shape and type, filled with zeros.
# set some default values
# 9000 rows with 20 features
# (8999, 20)
# print((int(ntrn0+ntrn1), 20)); 
# 8999
# print((int(ntrn0+ntrn1)))
xtrn = np.zeros((int(ntrn0+ntrn1), 20))
ytrn = np.zeros((int(ntrn0+ntrn1)))
# (8999, 20)
# print(xtrn.shape)
# (8999,)
# print(ytrn.shape)

xtrn[:ntrn0] = x0[:ntrn0]
xtrn[ntrn0:] = x1[:ntrn1]
# print(xtrn)

# One-dimensional slices
# The general syntax for a slice is array[start:stop:step]

# start from begining till end of vectors
ytrn[:ntrn0] = y0[:ntrn0]
# strong from vectors to end of labels
ytrn[ntrn0:] = y1[:ntrn1]
# print(ytrn)

# --- 5% for the validation set
# take class 0 vectors size bustract training set
n0 = int(x0.shape[0] - ntrn0)
# print(x0.shape[0])
# 896
# print(n0)
# take class 1 vectors size bustract training set
n1 = int(x1.shape[0] - ntrn1)
# print(x1.shape[0])
# 104
# print(n1)

# Set default values
# 500
# print(int(n0/2+n1/2))
xval = np.zeros((int(n0/2+n1/2), 20))
yval = np.zeros(int(n0/2+n1/2))

# set x vectors
# //: Divides the number on its left by the number on its right, rounds down the answer, and returns a whole number.
xval[:(n0//2)] = x0[ntrn0:(ntrn0+n0//2)]
# set x labels
xval[(n0//2):] = x1[ntrn1:(ntrn1+n1//2)]

# set y vectors
yval[:(n0//2)] = y0[ntrn0:(ntrn0+n0//2)]
# set y labels
yval[(n0//2):] = y1[ntrn1:(ntrn1+n1//2)]


# --- 5% for the TEST set
xtst = np.concatenate((x0[(ntrn0+n0//2):], x1[(ntrn1+n1//2):]))
ytst = np.concatenate((y0[(ntrn0+n0//2):], y1[(ntrn1+n1//2):]))

print("working")