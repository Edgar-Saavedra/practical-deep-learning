import numpy as np
from sklearn import decomposition

# we pass the pca object,original data, and the starting principal component index
def generateData(pca, x, start):
  # keep a copy  of the actual components
  original = pca.components_.copy()
  # number of principal component
  ncomp = pca.components_.shape[0]
  # call the forward transformation mapping the original
  # data along the principal components
  a = pca.transform(x)

  # the loop updates the two least important components by adding a random
  # value drawn from a normal curve with mean value 0 and standard deviation of 0.1
  # if the standard deviation is small then the new samples will be near the old samples
  # larger = farther away from old samples
  for i in range(start, ncomp):
    pca.components_[i, :] += np.random.normal(scale=0.1, size=ncomp)

  # call the inverse transformation 
  b = pca.inverse_transform(a)

  # return the new set of samples
  pca.components_ = original.copy()
  return b

def main():
  x = np.load("iris_features.npy")
  y = np.load("iris_labels.npy")

  N = 120
  x_train = x[:N]
  y_train = y[:N]

  x_test = x[N:]
  y_test = y[N:]

  pca = decomposition.PCA(n_components=4)
  pca.fit(x)
  print(pca.explained_variance_ratio_)
  
  # the first two components represent 97% of the variance
  # We'll leave the first two components alone
  start = 2
  # delcare the number of collections of samples.
  # the samples are based on the original data (150)
  nsets = 10

  # each new set will contain 150 samples (in this case)

  # save the original data
  nsamp = x_train.shape[0]

  # 9 new sets of sampels
  newx = np.zeros((nsets * nsamp, x_train.shape[1]))
  newy = np.zeros(nsets*nsamp, dtype="uint8")

  # loop to generate the new samples, on set of 150 at a time
  for i in range(nsets):
    # first pass copmies the original data
    if (i == 0):
      newx[0:nsamp, :] = x_train
      newy[0:nsamp] = y_train
    # remaining passes  update the source and destination indices
    else:
      newx[(i*nsamp):(i*nsamp+nsamp), :] = generateData(pca,x_train, start)
      newy[(i*nsamp):(i*nsamp+nsamp)] = y_train

  # scramble the order
  idx = np.argsort(np.random.random(nsets*nsamp))
  newx = newx[idx]
  newy = newy[idx]
  np.save("iris_train_features_augmented.npy", newx)
  np.save("iris_train_labels_augmented.npy", newy)
  np.save("iris_test_features_augmented.npy", x_test)
  np.save("iris_test_labels_augmented.npy", y_test)

main()