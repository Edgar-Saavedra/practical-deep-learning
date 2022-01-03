# how to augment the CIFAR-10 training set with random crops rotations and flips
import numpy as np
from PIL import Image

def augment(im, dim):
    #  get PIL image
    img = Image.fromarray(im)
    if (np.random.random() < 0.5):
      img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # rotating
    if (np.random.random() < 0.3333):
      z = (32-dim)/2
      r = 10 * np.random.random() - 5
      img = img.rotate(r, resample=Image.BILINEAR)
      # we center crop to we will not get any black pixels on the edges where the rotation
      # had no image formation to work with
      img = img.crop((z,z,32-z,32-z))
    else:
      # we want to crop more often than we rotate
      x = int((32-dim-1) * np.random.random())
      y = int((32-dim-1) * np.random.random())
      img = img.crop((x, y, x+dim, y+dim))
    # turn PIL image into array
    return np.array(img)

def main():
  # load existing dataset
  x = np.load("/Users/esaavedra/Work/personal/projects/practical-deep-learning/ch5/cifar10_train_images.npy")
  y = np.load("/Users/esaavedra/Work/personal/projects/practical-deep-learning/ch5/cifar10_train_labels.npy")
  # print('y', y)
  # define augmentation factor, crop size, center top
  factor = 10
  dim = 28
  z = (32-dim)/2

  # new image will be put in newx
  # there are 500,000;28;28;3 
  # there are 50,000 training images each with 32*32 pixels and 3 colors
  newx = np.zeros((x.shape[0] * factor, dim, dim, 3), dtype="uint8")
  newy = np.zeros(y.shape[0] * factor, dtype="uint8")

  # counter
  # fore every image in the old dataset, we will create nine completely new versions
  # and center crop the original
  k = 0

  for i in range(x.shape[0]):
    # easier to work with image form
    # fromarray turns a Numpy array into a PIL image
    im = Image.fromarray(x[i, :])
    im = im.crop((z, z, 32-z, 32-z))
    newx[k,...] = np.array(im)
    # just copy the label
    newy[k] = y[i]
    k += 1
    # we creating 9 versions of the current image
    for j in range(factor-1):
      newx[k, ...] = augment(x[i, :], dim)
      # just copy the label
      newy[k] = y[i]
      k += 1
  
  # scramble
  idx = np.argsort(np.random.random(newx.shape[0]))
  newx = newx[idx]
  newy = newy[idx]

  np.save("cifar10_aug_train_images.npy", newx)
  np.save("cifar10_aug_train_labels.npy", newy)

  x = np.load("cifar10_test_images.npy")
  # crop the original test set
  newx = np.zeros((x.shape[0], dim, dim, 3), dtype="uint8")

  for i in range(x.shape[0]):
    im = Image.fromarray(x[i, :])
    im = im.crop((z,z,32-z, 32-z))
    newx[i, ...] = np.array(im)
  
  np.save("cifar10_aug_test_images.npy", newx)
