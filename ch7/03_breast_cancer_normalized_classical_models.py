# 136
import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def run(training_vectors, training_labels, test_vectors, test_labels, model):
  model.fit(training_vectors, training_labels)
  print("score = %04f" % model.score(test_vectors, test_labels))

def main():
  # load data set
  # dataset is already randomized
  x = np.load("../ch5/breast_cancer_features_standard_normalized.npy")
  y = np.load("../ch5/breast_cancer_labels.npy")

  # keep 455 of 469 samples for training (80%) and the remaining 114 samples are the test set
  N = 455

  # split into training and test data
  x_train = x[:N]
  y_train = y[:N]

  x_test = x[N:]
  y_test = y[N:]

  print("Nearest Centroid")
  run(x_train, y_train, x_test,y_test, NearestCentroid())
  
  # K-NN requires more and more training samples as the number of features increases
  print("k-NN classifier (k=3):")
  run(x_train, y_train, x_test, y_test, KNeighborsClassifier(n_neighbors=3))
  
  print("k-NN classifier (k=7):")
  run(x_train, y_train, x_test, y_test, KNeighborsClassifier(n_neighbors=7))

  print("Naive Bayes classifier(Gaussian):")
  run(x_train, y_train, x_test, y_test, GaussianNB())

  # Different runs will lead to different trees since it will randomly select a feature and find the best split.
  print("Decision Tree Classifier")
  run(x_train, y_train, x_test, y_test, DecisionTreeClassifier())

  # Random forest classifeir is rand. We expect different results run to run.
  print("Random Forest classifier (estimator=5):")
  run(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators=5))

  print("Random Forest classifier (estimator=50):")
  run(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators=50))

  print("SVM (linear, C=1.0)")
  run(x_train, y_train, x_test, y_test, SVC(kernel="linear", C=1.0))

  #0.03125 since we have 32 features 1/32
  # SVM does use a random number generator so at times different runs will give different results
  print("SVM (RBF, C=1.0, gamma=0.03125):")
  run(x_train,y_train, x_test, y_test, SVC(kernel="rbf", C=1.0, gamma=0.03125))

main()