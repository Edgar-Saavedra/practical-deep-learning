# The MNIST data set consists of 28x28 pixel grayscale images of handwritten digits [0, 9]
# One digit centered per image.

# MNIST contains 60k training images, evenly split among the digits and 10k test images
# Since we have training images, we wont do k-fold

# We will use the vector form of the images.
# The images are unraveled so that the first 28 elements of the vector are row 0, the next 28 are row 1 and so on  for an input vector of 28 x 28 = 784 elements

# Images are stored as 8-bit grayscale images, data values range from 0 - 255

# We consider 3 versions: 
# 1. RAW: Raw byte versions
# 2. SCALED: we scale the data [0, 1] by dividing by 256 (number of possible grayscale values)
# 3. NORMALIZED: A normalized version where, per "feature" (pixels) we substract the mean of that features accross that datase and divid by the standard deviation
# 152
import time
import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import decomposition

def run(x_train, y_train, x_test, y_test, clf):
  start = time.time()
  clf.fit(x_train, y_train)
  execution_time_train = time.time() - start
  start = time.time()
  score = clf.score(x_test, y_test)
  execution_time_test = time.time() - start
  print("score = %0.4f (time, train=%8.3f, test=%8.3f)" % (score, execution_time_train, execution_time_test))

def train(x_train, y_train, x_test, y_test):
  print("Nearest Centroid : ", end='')
  run(x_train, y_train, x_test, y_test, NearestCentroid())

  print("k-NN classifier (k=3) : ", end='')
  run(x_train, y_train, x_test, y_test, KNeighborsClassifier(n_neighbors=3))

  print("k-NN classifier (k=7) : ", end='')
  run(x_train, y_train, x_test, y_test, KNeighborsClassifier(n_neighbors=7))

  print("Naive Bayes (Gaussian)", end='')
  run(x_train, y_train, x_test, y_test, GaussianNB())

  print("Decision Tree : ", end='')
  run(x_train, y_train, x_test, y_test, DecisionTreeClassifier())

  print("Random Forest (trees = 5) : ", end='')
  run(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators=5))

  print("Random Forest (trees = 50) : ", end='')
  run(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators=50))

  print("Random Forest (trees = 500) : ", end='')
  run(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators=500))

  print("Random Forest (trees = 1000) : ", end='')
  run(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators=1000))

  print("LinearSVM (C=0.01) : ", end='')
  run(x_train, y_train, x_test, y_test, LinearSVC(C=0.01))

  print("LinearSVM (C=0.1) : ", end='')
  run(x_train, y_train, x_test, y_test, LinearSVC(C=0.1))

  print("LinearSVM (C=1.0) : ", end='')
  run(x_train, y_train, x_test, y_test, LinearSVC(C=1.0))

  print("LinearSVM (C=10.0) : ", end='')
  run(x_train, y_train, x_test, y_test, LinearSVC(C=10.0))

def main():
  x_train = np.load("../ch5/mnist_train_vectors.npy").astype("float64")
  y_train = np.load("../ch5/mnist_train_labels.npy")
  x_test = np.load("../ch5/mnist_test_vectors.npy").astype("float64")
  y_test = np.load("../ch5/mnist_test_labels.npy")

  print("Models trained on raw[0, 255] images")
  train(x_train, y_train, x_test, y_test)

  print("Models trained on raw [0, 1) images:")
  # TODO: why 256?
  train(x_train/255.0, y_train, x_test/256.0, y_test)

  m = x_train.mean(axis=0)
  # TODO: WHAT DOES axis control?
  s = x_train.std(axis=0) + 1e-8

  # normalized
  x_ntrain = (x_train - m)/s
  x_ntest = (x_test - m)/s

  print("Models trained on normalized images:")
  train(x_ntrain, y_train, x_ntest, y_test)

  # TODO: why 15?
  pca = decomposition.PCA(n_components=15)
  pca.fit(x_ntrain)
  x_ptrain = pca.transform(x_ntrain)
  x_ptest = pca.transform(x_ntest)

  print("Models trained on first 15 PCA components of normalized images:")
  train(x_ptrain, y_train, x_ptest, y_test)

main()

# Models trained on raw[0, 255] images
# Nearest Centroid : score = 0.8203 (time, train=   0.087, test=   0.051)
# k-NN classifier (k=3) : score = 0.9705 (time, train=   0.012, test=   3.256)
# k-NN classifier (k=7) : score = 0.9694 (time, train=   0.012, test=   3.144)
# Naive Bayes (Gaussian)score = 0.5558 (time, train=   0.258, test=   0.175)
# Decision Tree : score = 0.8785 (time, train=  10.652, test=   0.007)
# Random Forest (trees = 5) : score = 0.9244 (time, train=   1.066, test=   0.015)
# Random Forest (trees = 50) : score = 0.9691 (time, train=  10.149, test=   0.089)
# Random Forest (trees = 500) : score = 0.9719 (time, train= 101.051, test=   0.829)
# Random Forest (trees = 1000) : score = 0.9714 (time, train= 205.697, test=   1.666)
# /Users/edgar/.pyenv/versions/3.9.15/lib/python3.9/site-packages/sklearn/svm/_base.py:1225: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
#   warnings.warn(
# LinearSVM (C=0.01) : score = 0.8759 (time, train= 105.918, test=   0.012)
# /Users/edgar/.pyenv/versions/3.9.15/lib/python3.9/site-packages/sklearn/svm/_base.py:1225: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
#   warnings.warn(
# LinearSVM (C=0.1) : score = 0.8472 (time, train= 105.098, test=   0.007)
# /Users/edgar/.pyenv/versions/3.9.15/lib/python3.9/site-packages/sklearn/svm/_base.py:1225: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
#   warnings.warn(
# LinearSVM (C=1.0) : score = 0.8835 (time, train= 101.844, test=   0.009)
# /Users/edgar/.pyenv/versions/3.9.15/lib/python3.9/site-packages/sklearn/svm/_base.py:1225: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
#   warnings.warn(
# LinearSVM (C=10.0) : score = 0.8685 (time, train= 101.858, test=   0.015)


# Models trained on raw [0, 1) images:
# Nearest Centroid : score = 0.8203 (time, train=   0.089, test=   0.013)
# k-NN classifier (k=3) : score = 0.9704 (time, train=   0.014, test=   3.185)
# k-NN classifier (k=7) : score = 0.9690 (time, train=   0.013, test=   3.060)
# Naive Bayes (Gaussian)score = 0.5551 (time, train=   0.291, test=   0.222)
# Decision Tree : score = 0.8780 (time, train=  10.626, test=   0.007)
# Random Forest (trees = 5) : score = 0.9240 (time, train=   1.035, test=   0.015)
# Random Forest (trees = 50) : score = 0.9674 (time, train=  10.167, test=   0.086)
# Random Forest (trees = 500) : score = 0.9710 (time, train= 102.249, test=   0.803)
# Random Forest (trees = 1000) : score = 0.9707 (time, train= 208.234, test=   1.640)
# LinearSVM (C=0.01) : score = 0.9172 (time, train=   3.768, test=   0.008)
# LinearSVM (C=0.1) : score = 0.9180 (time, train=  19.652, test=   0.017)
# /Users/edgar/.pyenv/versions/3.9.15/lib/python3.9/site-packages/sklearn/svm/_base.py:1225: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
#   warnings.warn(
# LinearSVM (C=1.0) : score = 0.9185 (time, train=  57.883, test=   0.009)
# /Users/edgar/.pyenv/versions/3.9.15/lib/python3.9/site-packages/sklearn/svm/_base.py:1225: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
#   warnings.warn(
# LinearSVM (C=10.0) : score = 0.9066 (time, train=  97.457, test=   0.008)



# Models trained on normalized images:
# Nearest Centroid : score = 0.8092 (time, train=   0.087, test=   0.020)
# k-NN classifier (k=3) : score = 0.9452 (time, train=   0.015, test=   3.405)
# k-NN classifier (k=7) : score = 0.9433 (time, train=   0.012, test=   3.504)
# Naive Bayes (Gaussian)score = 0.5239 (time, train=   0.229, test=   0.180)
# Decision Tree : score = 0.8775 (time, train=  10.861, test=   0.008)
# Random Forest (trees = 5) : score = 0.9196 (time, train=   1.067, test=   0.015)
# Random Forest (trees = 50) : score = 0.9668 (time, train=  10.583, test=   0.091)
# Random Forest (trees = 500) : score = 0.9711 (time, train= 102.386, test=   0.831)
# Random Forest (trees = 1000) : score = 0.9718 (time, train= 206.995, test=   1.626)
# /Users/edgar/.pyenv/versions/3.9.15/lib/python3.9/site-packages/sklearn/svm/_base.py:1225: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
#   warnings.warn(
# LinearSVM (C=0.01) : score = 0.9158 (time, train= 156.265, test=   0.013)
# /Users/edgar/.pyenv/versions/3.9.15/lib/python3.9/site-packages/sklearn/svm/_base.py:1225: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
#   warnings.warn(
# LinearSVM (C=0.1) : score = 0.9159 (time, train= 223.549, test=   0.009)
# /Users/edgar/.pyenv/versions/3.9.15/lib/python3.9/site-packages/sklearn/svm/_base.py:1225: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
#   warnings.warn(
# LinearSVM (C=1.0) : score = 0.9109 (time, train= 357.196, test=   0.007)
# /Users/edgar/.pyenv/versions/3.9.15/lib/python3.9/site-packages/sklearn/svm/_base.py:1225: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
#   warnings.warn(
# LinearSVM (C=10.0) : score = 0.8864 (time, train= 379.507, test=   0.008)


# Models trained on first 15 PCA components of normalized images:
# Nearest Centroid : score = 0.7520 (time, train=   0.007, test=   0.004)
# k-NN classifier (k=3) : score = 0.9353 (time, train=   0.040, test=   1.169)
# k-NN classifier (k=7) : score = 0.9369 (time, train=   0.024, test=   1.445)
# Naive Bayes (Gaussian)score = 0.8013 (time, train=   0.011, test=   0.005)
# Decision Tree : score = 0.8416 (time, train=   1.287, test=   0.002)
# Random Forest (trees = 5) : score = 0.8850 (time, train=   0.728, test=   0.008)
# Random Forest (trees = 50) : score = 0.9215 (time, train=   7.219, test=   0.073)
# Random Forest (trees = 500) : score = 0.9264 (time, train=  72.600, test=   0.730)
# Random Forest (trees = 1000) : score = 0.9251 (time, train= 145.762, test=   1.433)
# LinearSVM (C=0.01) : score = 0.8289 (time, train=   6.253, test=   0.012)
# /Users/edgar/.pyenv/versions/3.9.15/lib/python3.9/site-packages/sklearn/svm/_base.py:1225: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
#   warnings.warn(
# LinearSVM (C=0.1) : score = 0.8302 (time, train=  13.232, test=   0.022)
# /Users/edgar/.pyenv/versions/3.9.15/lib/python3.9/site-packages/sklearn/svm/_base.py:1225: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
#   warnings.warn(
# LinearSVM (C=1.0) : score = 0.8265 (time, train=  23.007, test=   0.010)
# /Users/edgar/.pyenv/versions/3.9.15/lib/python3.9/site-packages/sklearn/svm/_base.py:1225: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
#   warnings.warn(
# LinearSVM (C=10.0) : score = 0.7362 (time, train=  31.979, test=   0.020)