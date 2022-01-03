# Augmenting the Iris Dataset
#  Principal component analysis (PCA)
import numpy as np
import matplotlib.pylab as plt
from sklearn import decomposition

# keep first 2 features
def getFeatures():
  return np.load('/Users/esaavedra/Work/personal/projects/practical-deep-learning/ch5/iris_features.npy')[:, :2]

def getLabels():
  return np.load('/Users/esaavedra/Work/personal/projects/practical-deep-learning/ch5/iris_labels.npy')

def filterLabels(labels):
  return np.where(labels != 0)

# substract the per feature means
def substractPerfeatureMeans(features):
    features[:, 0] -= features[:, 0].mean()
    features[:, 1] -= features[:, 1].mean()
    return features

def fitPCAToData(features):
  # pca components available in components_ variable
  # copmonents are always listed in decreasing order.
  # The first component is the direction describing the majority of the variance.
  pca = decomposition.PCA(n_components = 2)
  pca.fit(features)
  return pca

def componentArrow(varianceFraction, axes, pca, index=0):
  componentX = varianceFraction[index] * pca.components_[index, 0]
  componentY = varianceFraction[index] * pca.components_[index, index]
  axes.arrow(0, 0, componentX, componentY, head_width=0.05, head_length=0.1, fc='r', ec='r')

def dummyPlot():
  x = [1,2,3,4,5,6,7,8]
  y = [4,1,3,6,1,3,5,2]

  plt.scatter(x,y)

  plt.title('Nuage de points avec Matplotlib')
  plt.xlabel('x')
  plt.ylabel('y')

  plt.savefig('ScatterPlot_01.png')
  plt.show()

  plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,
              cmap='viridis')
  plt.colorbar();  # show color scale

def draw(features, varianceFraction, pca):
  plt.scatter(features[:, 0], features[:, 1])
  ax = plt.axes()

  componentArrow(varianceFraction, ax, pca)
  componentArrow(varianceFraction, ax, pca, 1)

  plt.xlabel("$x_0$", fontsize=16)
  plt.ylabel("$x_1$", fontsize=16)
  plt.scatter(features[:, 0], features[:, 1], marker='o', color='black')
  plt.show()

def run():
  x = getFeatures()
  y = getLabels()
  idx = filterLabels(y)

  x = x[idx]
  x = substractPerfeatureMeans(x)
  pca = fitPCAToData(x)

  #  vector representing the fraction of the variance int the data explained
  # by each of the principal compnent directions
  # will have 2 components
  v = pca.explained_variance_ratio_

  draw(x, v, pca)

run()