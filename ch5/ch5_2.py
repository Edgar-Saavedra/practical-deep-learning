import numpy as np
import matplotlib.pyplot as plt

def getLabels(lines):
  # labels
  n = ["B", "M"]
  # labels
  return np.array([n.index(i.split(",")[1]) for i in lines], dtype="uint8")

def getFeatures(lines):
  # features
  return np.array([[float(j) for j in i.split(",")[2:]] for i in lines])

def randomize(labels):
  return np.argsort(np.random.random(labels.shape[0]))

def getData():
  with open("/Users/esaavedra/Work/personal/projects/practical-deep-learning/ch5/wdbc.data") as f:
    lines = [i[:-1] for i in f.readlines() if i != ""]
  return lines

def getStandardizedVersion(features):
  """Standardization/Normalizing should be done on most datasets (especially when using a traditionalmodel). So the features have 0 mean and standard ddeviation of 1"""
  # The mean of 0 and standard deviation of 1 usually applies to the standard normal distribution, 
  # often called the bell curve. The most likely value is the mean and it falls off as you get 
  # farther away. If you have a truly flat distribution then there is no value more likely than another.
  standard = (features - features.mean(axis=0)) / features.std(axis=0)
  print("getStandardizedVersion", standard)
  return standard

def save(features, features_standardized, labels):
  np.save("/Users/esaavedra/Work/personal/projects/practical-deep-learning/ch5/bc_features.npy", features)
  np.save("/Users/esaavedra/Work/personal/projects/practical-deep-learning/ch5/bc_features_standard.npy", features_standardized)
  np.save("/Users/esaavedra/Work/personal/projects/practical-deep-learning/ch5/bc_labels.npy", labels)
  
def plot(vector):
  if (type(vector).__module__ != np.__name__):
    vector = np.array(vector)
  plt.boxplot(vector)
  plt.show()

def run():
  lines = getData()
  labels = getLabels(lines)
  features = getFeatures(lines)
  random = randomize(labels)
  labels = labels[random]
  features = features[random]
  features_standard = getStandardizedVersion(features)
  save(features, features_standard, labels)
  plot(features_standard)