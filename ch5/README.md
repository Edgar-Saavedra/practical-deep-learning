## Objective
- Acquire raw data
- peprocess data (often most labor intensive)

## resources
- [iris repository](https://archive.ics.uci.edu/ml/index.php)
- [iris download](https://archive.ics.uci.edu/ml/datasets/iris), [iris.data](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)
- [Wisconsin Diagnostic Breast Cancer dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data)

# Steps
1. Randomize the data
2. Replace class names with integer labels
3. 5-1.py

## MNIST Data set
- use keras, but (cononical source)[yann.lecun.com/exdb/mnist]
- create additional dataset from this initial Keras set
  1. Unravel the images to form form feature vectors
  2. Permute the order of the images in the dataset. The reordering of the pixels will be deterministic and applied consitently across all images
  3. Create an unraveled feature vector version of the permuted images

## CIFAR-10 dataset
 - in Keras
 - [source page](https://www.cs.toronto.edu/%7Ekriz/cifar.html)
 - consists of 60,000 32X32 rgb images from
  - 10 classes
    - airplane
    - automobile
    - bird
    - cat
    - deer
    - dog 
    - frog
    - horse
    - ship
    - truck
  - 6000 samples in each class
  - Training set contains 50,000
  - TEst set contains 10,000