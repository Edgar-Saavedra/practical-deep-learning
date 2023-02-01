# PG 103
# Ugmented CIFAR-10 training set with radom corps, rotations and flips.
import numpy as np
from PIL import Image

# side note we must be sure that the Numpy array is a valid image data type like unsigned byte (uint8)


# method that takes in an image vector and dimensions and generates a new rendom augmentation rotating or flipping
def augment(im, dim):
  # Turn a numpy array representing an image into a PIL image
  img = Image.fromarray(im)
  if (np.random.random() < 0.5):
    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
  if (np.random.random() < 0.3333):
    z = (32 - dim) / 2
    r = 10 * np.random.random() - 5
    img = img.rotate(r, resample=Image.Resampling.BILINEAR)
    img = img.crop((z, z, 32-z, 32-z))
  else:
    # if no rotate then we crop
    x = int((32-dim-1) * np.random.random())
    y = int((32-dim-1) * np.random.random())
    img = img.crop((x, y, x+dim, y+dim))

  # Take a PIL image and turn it into a Numpy array
  return np.array(img)

def main():
  # load the existing dataset from ch_5_04_cifar_split_randomize.py
  x = np.load("cifar10_train_images.npy")
  y = np.load("cifar10_train_labels.npy")

  # define our augmentation factor, crop size and constatn for for defining center top
  factor = 10
  dim = 28
  z = (32 - dim)/2

  # THe new image will put into newx with te following dimensions (500,000; 28;28;3)
  newx = np.zeros((x.shape[0] * factor, dim, dim, 3), dtype="uint8")
  # this sets the training lable by the same factor
  newy = np.zeros(y.shape[0]*factor, dtype="uint8")

  # index into the new dataset
  k = 0

  # For every image in the old dataset we'll create nine completely new versions and center crop the original
  for i in range(x.shape[0]):
    # center crop the original
    im = Image.fromarray(x[i, :])
    im = im.crop((z,z,32-z,32-z))
    # Take a PIL image and turn it into a Numpy array
    newx[k, ...] = np.array(im)
    newy[k] = y[i]
    k += 1
  
    # create the nine new versions
    for j in range(factor-1):
      newx[k, ...] = augment(x[i, :], dim)
      newy[k] = y[i]
      k += 1
  

  idx = np.argsort(np.random.random(newx.shape[0]))
  newx = newx[idx]
  newy = newy[idx]
  np.save("cifar10_aug_train_images.npy", newx)
  np.save("cifar10_aug_train_labels.npy", newy)

  # create the croped test set
  x = np.load("cifar10_test_images.npy")
  newx = np.zeros((x.shape[0], dim, dim, 3), dtype="uint8")

  for i in range(x.shape[0]):
    im = Image.fromarray(x[i, :])
    im = im.crop((z, z, 32-z, 32-z))
    # Take a PIL image and turn it into a Numpy array
    newx[i, ... ] = np.array(im)
  
  np.save("cifar10_aug_test_images.npy", newx)

main()