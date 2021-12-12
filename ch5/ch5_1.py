import numpy as np
import matplotlib.pyplot as plt


def importData():
  with open("/Users/esaavedra/Work/personal/projects/practical-deep-learning/ch5/iris.data") as  f:
    lines = [i[:-1] for i in f.readlines()]

  n = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
  # loop through lines. split commas, get last line and its index from our map of labels

  # LABELS ---
  x = [n.index(i.split(",")[-1]) for i in lines if i != ""]
  # vectorize our labels
  x = np.array(x, dtype="uint8")

  # FEATURES ---
  # loop through lines, split on comma, exlude last item (label) get a floating point number from value
  y = [[float(j) for j in i.split(",")[:-1]] for i in lines if  i != ""]
  y = np.array(y)

  return [x, y]

def randomize(labels, feautures):
  # randomize our labels
  i = np.argsort(np.random.random(labels.shape[0]))
  print('labels before randomizing', labels)
  x = labels[i]
  print('labels after randomizing', x)
  print('feautures before randomizing', feautures)

  # NOTE STILL NEED MORE CLARITY AS TO HOW THIS WORKS with python numpy arrays
  y = feautures[i]
  print('y after randomizing', y)

  return [x, y]

def save(labels, features):
  np.save("/Users/esaavedra/Work/personal/projects/practical-deep-learning/ch5/iris_features.npy", features)
  np.save("/Users/esaavedra/Work/personal/projects/practical-deep-learning/ch5/iris_labels.npy", labels)


def plot(features):
  if (type(features).__module__ != np.__name__):
    features = np.array(features)
  plt.boxplot(features)
  plt.show()