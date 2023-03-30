# 148
# A Two Dimensional grid search for C and y for an RBF kernel SVM.

# We want to use k-fold validation
# there are 569 samples. 
# We want to split so thtere are a decent number of samples per fold
# This argues to making K small
# We also want to average out the effect of a bad split (larger k)

# If we set k = 5 we'll get a 113 sample per split
# This leaves a 80% for training and 20 percent for test for each combination of folds.
import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sys

# takes a train set, test set and model instance, trains the model and then socres the model against the test set
def run(train_set, train_labels, test_set, test_labels, model):
  model.fit(train_set, train_labels)
  return model.score(test_set, test_labels)

# take data set samples, the corresponding labels, the current fold number , and total number of folds
# deivid the full dataset in to m distinct number of folds
def split(dataset_samples, dataset_labels, current_k_fold_number, total_number_folds):
  # divide the full dataset in to number of folds
  number_samples_per_fold = int(dataset_labels.shape[0]/total_number_folds)

  # create a list of folds
  # each element of this list contains the feature vectors and labels of the fold
  list_of_folds = []

  for i in range(total_number_folds):
    # each element of this list contains the feature vectors and labels of the fold
    list_of_folds.append(
      [
        dataset_samples[(number_samples_per_fold * i):(number_samples_per_fold * i + number_samples_per_fold)],
        dataset_labels[(number_samples_per_fold * i):(number_samples_per_fold * i + number_samples_per_fold)]
      ]
    )

  # test set is the k-th fold so we set those values
  test_vectors, test_labels = list_of_folds[current_k_fold_number]


  train_vectors = []
  train_labels = []

  # take the remaining total number of folds - 1 and merge them into 
  # a new training set
  for i in range(total_number_folds):
    if (i == current_k_fold_number):
      continue
    else:
      a,b = list_of_folds[i]
      list_of_folds[i]
      train_vectors.append(a)
      train_labels.append(b)
  
  # when the loop ends training vectors is a list
  # each lement of which is a list representing the feature vectors
  # of the fold we want in the training set.
  # - first make a numpy array of the list
  # - reshape it so that training vectors has 30 columns (the number of features per vector) and number of samples per fold
  # - train vectors then becomes train_vectors minus the samples we put into the test fold
  # - we buld a train label so that the correct label goes with each feature in the train vectors
  train_vectors = np.array(train_vectors).reshape(((total_number_folds-1)*number_samples_per_fold,30))

  train_labels = np.array(train_labels).reshape((total_number_folds-1)*number_samples_per_fold)

  return [
    train_vectors,
    train_labels,
    test_vectors,
    test_labels
  ]

# pretty print function to show the per split scores
# it also shows the average score acorss all the splits
# The score is the overall accuracy of the model, 1 is perfection 0 is failure
def prettyPrint(vector, k_fold, label):
  m = vector.shape[1]
  print("%-19s: %0.4f +/- %0.4f | " % (label, vector[k_fold].mean(), vector[k_fold].std()/np.sqrt(m)), end='')
  for i in range(m):
      print("%0.4f " % vector[k_fold,i], end='')
  print()


def main():
  # load full dataset
  x = np.load("../ch5/breast_cancer_features.npy")
  y = np.load("../ch5/breast_cancer_labels.npy")

  # randomize ordering
  idx = np.argsort(np.random.random(y.shape[0]))
  
  x = x[idx]
  y = y[idx]

  # read number of folds
  m = 5

  # C -  fudge factor affects the size of the margin found
  # y (gamma) - this parameter relates to how spread out the Guassina Kernel is around a particular training point.
  Cs = np.array([0.01, 0.1, 1.0, 2.0, 10.0, 50.0, 100.0])

  # The first part is 1/30 the reciprocal of the number of features.
  # This is the default value for gamma (y) used in sklearn
  # We multiply this factor by an array (2^-4, 2^-3, 2^-2, 2^-1, 2^0, 2^1, 2^2, 2^3)
  gammaParameters = (1./30)*2.0**np.array([-4,-3, -2, -1, 0, 1, 2, 3])
  zmax = 0.0

  for C_Parameter in Cs:
    # search over
    # We iterate over all possible paris of C and Gamma , for each we do
    # five-fold validtion to get a set of five scores in z (vector
    for g in gammaParameters:
      z = np.zeros(m)
      for k in range(m):
        x_train, y_train, x_test, y_test = split(x, y, k, m)
        z[k] = run(x_train, y_train, x_test, y_test, SVC(C=C_Parameter, gamma=g, kernel="rbf"))
      
      # we then as if the mean of this set is greater than the current maximum
      if (z.mean() > zmax):
        zmax = z.mean()
        bestC = C_Parameter
        bestg = g

  print("best C = %0.5f"%bestC)
  print("gamma C = %0.5f"%bestg)
  print("accuracy = %0.5f"%zmax)

main()
