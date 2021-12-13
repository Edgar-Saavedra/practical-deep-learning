import numpy as np
from tensorflow import keras
from keras.datasets import cifar10

trn_img_count = 50000
tst_img_count = 10000
img_size = 32
color_dimensions_RGB = 3
path = "/Users/esaavedra/Work/personal/projects/practical-deep-learning/ch5/"

def getCifarData():
  (features_trn, labels_trn), (features_tst, labels_tst) = cifar10.load_data()

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



def randomize(dictionary):
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

def saveData(dictionary):
  np.save(f"{path}cifar10_train_images.npy", dictionary["train"]["features"])
  np.save(f"{path}cifar10_train_labels.npy", dictionary["train"]["labels"])
  np.save(f"{path}cifar10_test_images.npy", dictionary["test"]["features"])
  np.save(f"{path}cifar10_test_labels.npy", dictionary["test"]["labels"])

def getUnraveledVectors(dictionary):
  features_trn_unravel = dictionary["train"]["features"].reshape((trn_img_count, img_size * img_size * color_dimensions_RGB))
  features_tst_unravel = dictionary["test"]["features"].reshape((tst_img_count, img_size * img_size * color_dimensions_RGB))
  return {
    "train": {
      "features": features_trn_unravel
    },
    "test": {
      "features": features_tst_unravel
    }
  }

def saveUnraveledVectors(dictionary):
  np.save(f"{path}cifar10_train_vectors.npy", dictionary["train"]["features"])
  np.save(f"{path}cifar10_test_vectors.npy", dictionary["test"]["features"])

def run():
  data = getCifarData()
  data = randomize(data)
  saveData(data)

  unravel = getUnraveledVectors(data)
  saveUnraveledVectors(unravel)
