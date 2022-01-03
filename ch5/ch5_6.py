import numpy as np
from sklearn import decomposition

def generateData(pca, x, start):
  original = pca.components_.copy()
  # number of principal components
  ncomp = pca.components_.shape[0]
  # call forward transformation mapping the
  # original data along the principal components
  a = pca.transform(x)

  # loop updates the 2 least importnat componentsby adding
  # a random value drawn from a normal curve with mean value 0
  # and a stardard deviation of 0.1.
  # 0.1 cuz if the deviation is small then the new samples will 
  # be near the old samples if larger they will be farther away 
  for i in range(start, ncomp):
    pca.components_[i, :] += np.random.normal(scale=0.1,size=ncomp)
  # call the inverse tranformation
  b = pca.inverse_transform(a)
  # return new set of sampels
  pca.components_ = original.copy()
  return b

def main():
  # load iris data
  x = np.load("/Users/esaavedra/Work/personal/projects/practical-deep-learning/ch5/iris_features.npy")
  # labels
  y = np.load("/Users/esaavedra/Work/personal/projects/practical-deep-learning/ch5/iris_labels.npy")

  N = 120
  x_train = x[:N]
  y_train = y[:N]

  x_test = x[N:]
  y_test = y[N:]

  pca = decomposition.PCA(n_components=4)
  pca.fit(x)

  print(pca.explained_variance_ratio_)
  # first 2 principal components describe over 97% of the variance
  start = 2

  # set means a new collections of samples
  # each new set will cointain 150 samples **type is 150 should be 120**
  nsets = 10
  # this returns 120 however
  nsamp = x_train.shape[0]
  print('x_train.shape[0]', x_train.shape[0])
  print('x_train.shape[1]', x_train.shape[1])
  newx = np.zeros((nsets * nsamp, x_train.shape[1]))
  newy = np.zeros(nsets * nsamp, dtype="uint8")

  # loop to generate new samples, on set of 150 (120) at a time
  for i in range(nsets):
    # copy the original data into the ouput arrays
    if (i == 0):
      newx[0:nsamp, :] = x_train
      newy[0:nsamp] = y_train
    else:
      # update the source and destination indices of the ouput arrays
      # appropriately, but instead assignin x assign the out pout of generateData() whe loop is done
      newx[(i*nsamp):(i * nsamp + nsamp), :] = generateData(pca, x_train, start)
      newy[(i*nsamp):(i * nsamp + nsamp)] = y_train
  
  # scramble the order of the entire dataset and write it to disk.
  idx = np.argsort(np.random.random(nsets*nsamp))
  newx = newx[idx]
  newy = newy[idx]

  np.save("iris_train_features_augmented.npy", newx)
  np.save("iris_train_labels_augmented.npy", newy)
  np.save("iris_test_features_augmented.npy", x_test)
  np.save("iris_test_labels_augmented.npy", y_test)

main()