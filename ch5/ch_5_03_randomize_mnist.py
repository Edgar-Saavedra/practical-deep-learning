import numpy as np
import keras
from keras.datasets import mnist

# x - labels
# y - vectors

#  load keras
# The dataset is stored in four numpy arrays.
# xtn - (60000, 28, 28) for 60000 training images
# ytrn - has integer [0, 9] 
# xtst - 10000 test images
# ytst - with label
(xtrn, ytrn), (xtst, ytst) = mnist.load_data()
idx = np.argsort(np.random.random(ytrn.shape[0]))
xtrn = xtrn[idx]
ytrn = ytrn[idx]

idx = np.argsort(np.random.random(ytst.shape[0]))
xtst = xtst[idx]
ytst = ytst[idx]

np.save("mnist_train_images.py", xtrn)
np.save("mnist_train_labels.npy", xtrn)

np.save("mnist_test_images.py", xtst)
np.save("mnist_test_labels.py", ytst)

# unravel - to form feature vectors so that we can use this dataset with traditional models
# We unravel the training and test images and turn them into vectors of 784 elements
# it unravels the images so they are placed end-to-end 28*28 = 784
xtrnv = xtrn.reshape((60000, 28*28))
xtstv = xtst.reshape((10000, 28*28))

np.save("mnist_train_vectors.npy", xtrnv)
np.save("mnist_test_vectors.npy", xtstv)

# generate permuntation - the pixels will no longer be in the order that produces the digit image.
# the reordering will be deterministic and applied consitently across all images
idx = np.argsort(np.random.random(28*28))
for i in range(60000):
  xtrnv[i,:] = xtrnv[i, idx]
for i in range(10000):
  xtstv[i,:] = xtstv[i, idx]

np.save("mnist_train_scramble_vectors.npy", xtrnv)
np.save("mnist_test_scramble_vectors.npy", xtstv)

# Form new scrambled feature vector images of permuted images
# The scrambled images are made from the vectors 
t = np.zeros((60000, 28, 28))
for i in range(60000):
  t[i,:,:] = xtrnv[i, :].reshape((28,28))
np.save("mnist_train_scrambled_images.npy", t)

t = np.zeros((10000, 28, 28))

for i in range(10000):
  t[i, :, :] = xtstv[i, :].reshape((28,28))
np.save("mnist_test_scrambled_images.npy", t)