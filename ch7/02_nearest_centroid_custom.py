import numpy as np

# return the centroid for 3 classes
def centroids(x, y):
  c0 = x[np.where(y==0)].mean(axis=0)
  c1 = x[np.where(y==1)].mean(axis=0)
  c2 = x[np.where(y==2)].mean(axis=0)
  return [c0, c1, c2]

def predict(c0, c1, c2, x):
  # dfine the vector of predictions
  p = np.zeros(x.shape[0], dtype="uint8")
  for i in range(x.shape[0]):
    # x[i] current sample
    # one prediction per sample
    # index of the smalles distance
    # we set the list of 3 values
    # c0-x[i] retursn a vector of 4 numbers because we have 4 features.
    # this squared vector is summed element by element to return the distance measuer
    # since we are simply looking for the smalles distance to each of the centroids we dont need
    # to calculate the square root. The smalles value will still be the the smallest value.
    d = [
      ((c0-x[i])**2).sum(),
      ((c1-x[i])**2).sum(),
      ((c2-x[i])**2).sum(),
    ]
    # predicted class label
    p[i] = np.argmin(d)
  return p

def main():
  # load data set
  x = np.load("../ch5/iris_features.npy")
  y = np.load("../ch5/iris_labels.npy")

  N = 120

  x_train = x[:N]
  y_train = y[:N]

  x_test = x[N:]
  y_test = y[N:]

  c0, c1, c2 = centroids(x_train, y_train)

  p = predict(c0, c1 , c2, x_test)

  nc = len(np.where(p == y_test)[0])
  nw = len(np.where(p != y_test)[0])

  acc = float(nc) / (float(nc) + float(nw))

  print("predicted:", p)
  print("actual : ", y_test)
  print("test accuracy = %0.4f" % acc)

main()