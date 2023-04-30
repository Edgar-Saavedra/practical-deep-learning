# ------------------------------------------------
# Training for custom NN
# ------------------------------------------------
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier

# 1 load the training and testing data from disk
xtrain = np.load("iris2_train.npy")
ytrain = np.load("iris2_train_labels.npy")
xtest = np.load("iris2_test.npy")
ytest = np.load("iris2_test_labels.npy")

# 1 setup the NN object
# hidden_layer_sizes : network has 2 hidden layers, the first with 3 nodes and second with 2 nodes
# activation: The network is also using logistic layers (this is another name for a sigmoid layer)
clf = MLPClassifier(
  hidden_layer_sizes=(3,2),
  activation="logistic",
  solver="adam",
  tol=1e-9,
  max_iter=5000,
	# let us know the results of each itteration
  verbose=True
)

# we call fit to train the model
clf.fit(xtrain, ytrain)
# gives us the output probabilities on the test data
prob = clf.predict_proba(xtest)
# calculate the score over the test set as we have done before
score = clf.score(xtest, ytest)

# We want to store the learned weights and biases
w12 = clf.coefs_[0]
w23 = clf.coefs_[1]
w34 = clf.coefs_[2]

b1 = clf.intercepts_[0]
b2 = clf.intercepts_[1]
b3 = clf.intercepts_[2]

# save them
weights = [w12, b1, w23, b2, w34, b3]
pickle.dump(weights, open("iris2_weights.pkl", "wb"))

print()
print("Test Results:")
print("  Overall score: %0.7f" % score)
print()

for i in range(len(ytest)):
  p = 0 if (prob[i,1] < 0.5) else 1
  print("%03d: %d - %d, %0.7f" % (i, ytest[i], p, prob[i, 1]))
