# Implementing a simple NN
# We will implement the sample NN and train it on 2 features from the iris dataset.
# We will implement the NN from scratch but use sklearn to train it.

# Goal: see how straightforward it is to implement a simple NN.

# This network accepts an imput feature vector with 2 features.
# It has 2 hidden layers - one with 3 nodes and othe other with 2 nodes
# It has 1 sigmoid output.
# The activation function of the hidden nodes are also sigmoids.

# Building the dataset.

import numpy as np

d = np.load("../ch5/iris_train_features_augmented.npy")
l = np.load("../ch5/iris_train_labels_augmented.npy")

# 1 we want only class 1 + 2
d1 = d[np.where(l==1)]
d2 = d[np.where(l==2)]

a = len(d1)
b = len(d2)

# 2 we're keeping only features 2 and 3 and put them in x
x = np.zeros((a+b, 2))

x[:a, :] = d1[:, 2:]
x[a:, :] = d2[:, 2:]

# 3 We build the labels y
# we recode the class labels 0 and 1
y = np.array([0]*a+[1]*b)
# scramble the order of the samples 
i = np.argsort(np.random.random(a+b))
x = x[i]
y = y[i]

# 4 and write new dataset to disk
np.save("iris2_train.npy", x)
np.save("iris2_train_labels.npy", y)

# 5 we repeate this process to build the test samples
d = np.load("../ch5/iris_test_features_augmented.npy")
l = np.load("../ch5/iris_test_labels_augmented.npy")

d1 = d[np.where(l==1)]
d2 = d[np.where(l==2)]

a = len(d1)
b = len(d2)

x = np.zeros((a+b, 2))
x[:a, :] = d1[:, 2:]
x[a:, :] = d2[:, 2:]

y = np.array([0]*a+[1]*b)
i = np.argsort(np.random.random(a+b))
x = x[i]
y = y[i]
np.save("iris2_test.npy", x)
np.save("iris2_test_labels.npy", y)
