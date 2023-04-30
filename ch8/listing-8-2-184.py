# ------------------------------------------------
# Using trained weights and biases to classify test samples for custom NN
# ------------------------------------------------

# We'll assume that it's already trained and we already know the weights and biases.

import numpy as np
import pickle
import sys

# numpy does not have the sigmoid natively
def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

# Implements the network
def evaluate(x, y, w):
	# pull the individual weight matrices, and bias vectors from the supplied weight list  (w)
	# w12 is a 2x3 matrix mapping 2 the two element INPUT to the first (1) hidden layer with 3 nodes
	# w23 is  a 3x2 matrix mapping the first(1) HIDDEN layer to the second(2) HIDDEN layer.
	# w34 is a 2x1 matrix maping the second(2) hidden layer to the OUTPUT
	# Bias vectors are b1 (3 elements), b2 (2 elements), b3 (single element).
	# weight matrices are not of the same shape as we indicated they would be. They are transposes. Scalar multiplication is commutative meaning ab = ba
	w12, b1, w23, b2, w34, b3 = w

	# evalut the sets the number correct (nc) & number wrong (nw)
	nc = nw = 0
	# define prob a vector to hold the output probability value for each of the test samples.
	prob = np.zeros(len(y))
	
	# Loop applies to the entire network to each test sample.
	for i in range(len(y)):
		# First we map the input vector to the first hidden layer, a vector of 3 number, the activation for each of the 3 hidden nodes
		a1 = sigmoid(np.dot(x[i], w12) + b1)
		# We then take the first hidden layer activations and calculate the second hidden layer. This is a 2 element vector as there are 2 nodes in the second hidden layer
		a2 = sigmoid(np.dot(a1, w23) + b2)
		# calculate the output value for the current input vector and store it in the prob array
		prob[i] = sigmoid(np.dot(a2, w34) + b3)

		# Check the output value of the network if less than or not of 0.5.
		z = 0 if prob[i] < 0.5 else 1

		# increment the correct (nc) or incorect (nw) coutners base ond the actual label for this sample y[i]
		if (z == y[i]):
			nc += 1
		else:
			nw += 1
	# overall accuracy is returned as the number of correctly classifeid samples divided by total number of samples 
	return [float(nc)/ float(nc + nw), prob]

# load the test samples and labels
# xtest is of shape 23 x 2 we have 23 test samples with 2 features
# ytest is a vector of 23 labels
xtest = np.load("iris2_test.npy")
ytest = np.load("iris2_test_labels.npy")

# when we train the the network we will store the weights and biases as a list of numpy arrays
# weights has 6 elements representing the 3 weight matrices and 3 bias vectors that define the network
# these are magic numbers that our training has conditioned the the dataset
weights = pickle.load(open("iris2_weights.pkl", "rb"))

# we call evaluate for each of the test samples throughout the network
# returns the socre, and probabilities for each sample (prob), predicted label and associated output probability of beaing class 1
score, prob = evaluate(xtest, ytest, weights)

print()

for i in range(len(prob)):
	print("%3d: actual: %d predict: %d prob: %0.7f" % (i, ytest[i], 0 if (prob[i] < 0.5) else 1, prob[i]))

# accuracy score is printed
print("Score = %0.4f" % score)
