import numpy as np
from PIL import Image

def exampleConcat(path, name, width = 28, max_height=10000, columns=100):
  x = np.load(path)
  concat_width = columns * width
  concat_width = int(concat_width)
  print('x.shape[0]', x.shape[0])
  concat_height = (x.shape[0] / columns) * width
  concat_height = int(concat_height)
  if (concat_height > max_height):
    concat_height = max_height

  dst = Image.new('RGB', (concat_width, concat_height))

  y_cord = 0

  for key in range(x.shape[0]):
      if (key % columns == 0):
        y_cord = key/columns * width;
        y_cord = int(y_cord)
        if (y_cord < concat_height):
          img = Image.fromarray(x[key])
          dst.paste(img, (0, y_cord))
      else :
        x_cord = key%columns * width;
        x_cord = int(x_cord)
        if (y_cord < concat_height):
          img = Image.fromarray(x[key])
          dst.paste(img, (x_cord, y_cord))

  dst.save(f"concat_example_{name}.png", format="png")

def main():
  exampleConcat("/Users/esaavedra/Work/personal/projects/practical-deep-learning/ch5/cifar10_aug_test_images.npy", "test")
  exampleConcat("/Users/esaavedra/Work/personal/projects/practical-deep-learning/ch5/cifar10_aug_train_images.npy", "train")