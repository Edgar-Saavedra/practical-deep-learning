import numpy as np
from tensorflow import keras
from keras.datasets import mnist

trn_img_count = 60000
tst_img_count = 10000
img_size = 28
path = "/Users/esaavedra/Work/personal/projects/practical-deep-learning/ch5/"

def getMnistData():
  (features_trn, labels_trn), (features_tst, labels_tst) = mnist.load_data()

  return {
    "train": {
      "features": features_trn,
      "labels": labels_trn
    },
    "test": {
      "features": features_tst,
      "labels": labels_tst
    }
  }

def randomizeMnist(dictionary):
  random = np.argsort(np.random.random(dictionary["train"]["labels"].shape[0]))
  features_trn = dictionary["train"]["features"][random]
  labels_trn = dictionary["train"]["features"][random]

  random = np.argsort(np.random.random(dictionary["test"]["labels"].shape[0]))
  features_tst = dictionary["test"]["features"][random]
  labels_tst = dictionary["test"]["labels"][random]

  return {
    "train": {
      "features": features_trn,
      "labels": labels_trn
    },
    "test": {
      "features": features_tst,
      "labels": labels_tst
    }
  }

def getUnraveledVectors(dictionary):
  features_trn_unravel = dictionary["train"]["features"].reshape((trn_img_count, img_size * img_size))
  features_tst_unravel = dictionary["test"]["features"].reshape((tst_img_count, img_size * img_size))
  return {
    "train": {
      "features": features_trn_unravel
    },
    "test": {
      "features": features_tst_unravel
    }
  }

def saveMnistData(dictionary):
  np.save(f"{path}mnist_train_images.npy", dictionary["train"]["features"])
  np.save(f"{path}mnist_train_labels.npy", dictionary["train"]["labels"])
  np.save(f"{path}mnist_test_images.npy", dictionary["test"]["features"])
  np.save(f"{path}mnist_test_labels.npy", dictionary["test"]["labels"])

def saveMnistUnraveledVectors(dictionary):
  np.save(f"{path}mnist_train_vectors.npy", dictionary["train"]["features"])
  np.save(f"{path}mnist_test_vectors.npy", dictionary["test"]["features"])


def saveMnistScrambledVectors(dictionary):
  np.save(f"{path}mnist_train_scrambled_vectors.npy", dictionary["train"]["features"])
  np.save(f"{path}mnist_test_scrambled_vectors.npy", dictionary["test"]["features"])

def getUndoUnravelImages(vector, count, img_size):
  t = np.zeros((count, img_size, img_size))
  for i in range(count):
    t[i, :, :] = vector[i, :].reshape((img_size, img_size))

def saveMnisUndoUnraveledImages(dictionary):
  features_trn = dictionary["train"]["features"]
  features_tst = dictionary["test"]["features"]

  t = getUndoUnravelImages(features_trn, trn_img_count, img_size)
  np.save(f"{path}mnist_train_scrambled_images.npy", t)

  t = getUndoUnravelImages(features_tst, tst_img_count, img_size)
  np.save(f"{path}mnist_test_scrambled_images.npy", t)


def scrambelUnraveled(dictionary):
  features_trn_unravel = dictionary["train"]["features"]
  features_tst_unravel = dictionary["test"]["features"]
  idx = np.argsort(np.random.random(img_size * img_size))

  for i in range(trn_img_count):
    features_trn_unravel[i,:] = features_trn_unravel[i, idx]

  for i in range(tst_img_count):
    features_tst_unravel[i, :] = features_tst_unravel[i, idx]
  
  return {
    "train": {
      "features": features_trn_unravel
    },
    "test": {
      "features": features_tst_unravel
    }
  }

def run():
  mnist = getMnistData()
  random = randomizeMnist(mnist)
  saveMnistData(random)
  unraveled = getUnraveledVectors(random)
  saveMnistUnraveledVectors(unraveled)
  scrambeled = scrambelUnraveled(unraveled)
  saveMnistScrambledVectors(scrambeled)
  saveMnisUndoUnraveledImages(scrambeled)
  # NOTE no need to standardize image. They are all on the same scale already, since they are pixels
