import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# sklearn has uniform interface
# we can simplify things by using the same function to train and test
def run(train_vectors, train_labels, test_vectors, test_labels, modelClass):
  # fit the model to the data, this is the training step
  modelClass.fit(train_vectors, train_labels)
  # test how well it does by calling predict
  print("predictions :", modelClass.predict(test_vectors))
  print("actual labels:", test_labels)
  print("score = %0.4f" % modelClass.score(test_vectors, test_labels))
  print()

def main():
  # load data
  x = np.load("../ch5/iris_features.npy")
  y = np.load("../ch5/iris_labels.npy")

  # we hold back 30 samples
  N = 120

  # seperate in to train
  x_train = x[:N]
  y_train = y[:N]

  # seperate to test
  x_test = x[N:]
  y_test = y[N:]

  # Load augmented data set
  xa_train = np.load("../ch5/iris_train_features_augmented.npy")
  ya_train = np.load("../ch5/iris_train_labels_augmented.npy")

  xa_test = np.load("../ch5/iris_test_features_augmented.npy")
  ya_test = np.load("../ch5/iris_test_labels_augmented.npy")


  print("Nearest Centriod:")
  run(x_train, y_train, x_test, y_test, NearestCentroid())

  print("k-NN Classifier (k=3):")
  run(x_train, y_train, x_test,y_test, KNeighborsClassifier(n_neighbors=3))

  # this is the correct way to use to e Naive Bayes classifier if there are continuous values
  print("Naive Bayes classifier (Gaussian):")
  run(x_train, y_train, x_test, y_test, GaussianNB())

  # Assumes the features a selected form a discrete set of possible values
  print("Naive Bayes classifier (Multinomial):")
  run(x_train, y_train, x_test, y_test, MultinomialNB())

  # 
  print("Decision Tree Classifier:")
  run(x_train, y_train, x_test, y_test, DecisionTreeClassifier())

  # 5 estimators are used, meaning 5 random trees are created and trained
  print("Random Forest classifier (estimator=5), augmented:")
  run(xa_train, ya_train, xa_test, ya_test, RandomForestClassifier(n_estimators=5))

  print("SVM (linear, C=1.0), augmented:")
  run(xa_train, ya_train, xa_test, ya_test, SVC(kernel="linear", C=1.0))

  print("SVM (RBF, C=1.0, gamma=0.25), augmented:")
  run(xa_train, ya_train, xa_test, ya_test, SVC(kernel="rbf", C=1.0, gamma=0.25))

  print("SVM (RBF, C=1.0, gamma=0.001, augmented)")
  run(xa_train, ya_train, xa_test, ya_test, SVC(kernel="rbf", C=1.0, gamma=0.001))

  # This shows that data augmentation turns a lousy classifier into a good one.
  print("SVM (RBF, C=1.0, gamma=0.001, original")
  run(x_train, y_train, x_test, y_test, SVC(kernel="rbf", C=1.0, gamma=0.001))

main()