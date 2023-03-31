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