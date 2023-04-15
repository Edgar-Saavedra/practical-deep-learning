# Plotting of 
import numpy as np
import matplotlib.pyplot as plt

def main():
  total_variance = np.load("mnist_pca_tv.npy")
  naive_bayes = np.load("mnist_pca_nb.npy")
  random_forest = np.load("mnist_pca_rf.npy")
  pca = np.load("mnist_pca_sv.npy")

  print("total_variance", total_variance.shape)
  print("naive_bayes", naive_bayes.shape)
  print("random_forest", random_forest.shape)
  print("pca", pca.shape)

  plt.plot(total_variance[0], total_variance[1])
  plt.plot(naive_bayes[0], naive_bayes[1])
  plt.plot(random_forest[0], random_forest[1])
  plt.plot(pca[0], pca[1])
  plt.show()

main()