# SCALING DATA
# This file shows how one can create Validation, Training datas from 2 classes and create splits.
# 70

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

# make_classification is a method to create a split of classes based on number of samples.
from sklearn.datasets import make_classification

a,b = make_classification(n_samples=10000, weights=(0.9,0.1))

# https://numpy.org/doc/stable/reference/arrays.indexing.html
# split into class 0 and class 1 collections
# class 0 
idx = np.where(b == 0)[0]
# features
x0 = a[idx, :]
# classes
y0 = b[idx]

# class 1
idx = np.where(b == 1)[0]
# features
x1 = a[idx, :]
# classes
y1 = b[idx]

# randomize ordering Class 0
idx = np.argsort(np.random.random(y0.shape))
x0 = x0[idx] #features
y0 = y0[idx] #labels

# randomize ordering Class 1
idx = np.argsort(np.random.random(y1.shape))
x1 = x1[idx] #features
y1 = y1[idx] #labels

# --- TRAINING SET ----
# extrat first 90 percent of samples for the two classes and build
# the training subset with samples in xtrn and labels in ytrn
# class 0
ntrn0 = int(0.9*x0.shape[0])
# class 1
ntrn1 = int(0.9*x1.shape[0])

# samples (features)
xtrn = np.zeros((int(ntrn0+ntrn1), 20))
# combine trainign set with class 0 and class 1
xtrn[:ntrn0] = x0[:ntrn0]
# print(x0[:ntrn0].shape)
# append class 1 to end
xtrn[ntrn0:] = x1[:ntrn1]
# print(x1[:ntrn1].shape)

# labels
# combine trainign set with class 0 and class 1 labels
ytrn = np.zeros((int(ntrn0+ntrn1))) #labels are in 1 row
ytrn[:ntrn0] = y0[:ntrn0]
ytrn[ntrn0:] = y1[:ntrn1]
# --- END: TRAINING SET ----

# --- VALIDATION SET ----
# Get remaining 5 precent for class 0 validation
n0 = int(x0.shape[0] - ntrn0)
# for class 1 validation
n1 = int(x1.shape[0] - ntrn1)


xval = np.zeros((int(n0/2+n1/2), 20)) #samples
yval = np.zeros(int(n0/2+n1/2)) #labels

# add class 0 samples and append class 1 **samples**.
xval[:(n0//2)] = x0[ntrn0:(ntrn0+n0//2)] 
xval[(n0//2):] = x1[ntrn1:(ntrn1+n1//2)]

# add class 0 samples and append class 1 **labels**.
yval[:(n0//2)] = y0[ntrn0:(ntrn0+n0//2)]
yval[(n0//2):] = y1[ntrn1:(ntrn1+n1//2)]
# --- END: VALIDATION SET ----

# --- TEST SET ---
# Last5 percent for Test set
# class 0 and 1 samples
xtst = np.concatenate((x0[(ntrn0+n0//2):], x1[(ntrn1+n1//2):]))
# class 0 and 1 labels
ytst = np.concatenate((y0[(ntrn0+n0//2):], y1[(ntrn1+n1//2):]))
# --- END: TEST SET ---