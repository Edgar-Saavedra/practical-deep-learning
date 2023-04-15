# 158
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
  return (score, execution_time_train, execution_time_test)

def main():
  # load dtaset
  x_train = np.load("../ch5/mnist_train_vectors.npy").astype("float64")
  y_train = np.load("../ch5/mnist_train_labels.npy")
  x_test = np.load("../ch5/mnist_test_vectors.npy").astype("float64")
  y_test = np.load("../ch5/mnist_test_labels.npy")

  # compute normalized versions
  m = x_train.mean(axis=0)
  s = x_train.std(axis=0) + 1e-8

  x_ntrain = (x_train - m) / s
  x_ntest = (x_test - m) / s

  n = 2
  # setup storage for the number of PCA components - from 10 to 780 in steps of 10
  pcomp = np.linspace(10,780, n, dtype="int16")

  #  Naive Bayes model: store naiveBayes
  nb = np.zeros((n,4))
  # Random Forest model: store Random Forest
  rf = np.zeros((n,4))
  # SVC model: store Liner SV
  sv = np.zeros((n,4))
  # Total variance: store amount of variance
  tv = np.zeros((n,2))

  # loop over pca components
  for i,p in enumerate(pcomp):
    pca = decomposition.PCA(n_components=p)
    pca.fit(x_ntrain)

    xtrain = pca.transform(x_ntrain)
    xtest = pca.transform(x_ntest)
  
    # explained_variance_ratio_ : ndarray of shape (n_components,)
    # Percentage of variance explained by each of the selected components.
    tv[i, :] = [p, pca.explained_variance_ratio_.sum()]
  
    sc, etrn, etst = run(xtrain, y_train, xtest, y_test, GaussianNB())
    print("GaussianNB score = %0.4f (time, train=%8.3f, test=%8.3f)" % (sc, etrn, etst))
    nb[i,:] - [p, sc, etrn, etst]

    sc, etrn, etst = run(xtrain, y_train, xtest, y_test, RandomForestClassifier(n_estimators=50))
    print("RandomForestClassifier score = %0.4f (time, train=%8.3f, test=%8.3f)" % (sc, etrn, etst))
    rf[i, :] = [p,sc, etrn, etst]

    sc, etrn, etst = run(xtrain, y_train, xtest, y_test, LinearSVC(C=1.0))
    print("LinearSVC score = %0.4f (time, train=%8.3f, test=%8.3f)" % (sc, etrn, etst))
    sv[i,:] = [p, sc, etrn, etst]

  # Total variance: we will use these values for plotting latter and store them
  np.save("mnist_pca_tv.npy", tv)
  #  Naive Bayes model: Used to plot the accuracy of the Naive Bayes model 
  np.save("mnist_pca_nb.npy", nb)
  # Random Forest model: Used to plot the accuracy of the Random Forest model 
  np.save("mnist_pca_rf.npy", rf)
  # SVC model: Used to plot the accuracy of the Linear SVC model 
  np.save("mnist_pca_sv.npy", sv)


main()