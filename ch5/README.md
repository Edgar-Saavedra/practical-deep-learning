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

See page 86
1. Analyze the data, understand the structure
2. Standardize - features a re different scales 
3. Sort - hold some back and categorize.

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

## DATA Augmentation
- We need Models to **interpolate**
- **Data Augmentation** should be used whereever its feasible and small dataset
  - Take data we already have and modify it to create a new sample that could have plausibly come from the same parent distribution
- **Curse Of Dimensionality** (too many features) its solution is to fill in the space of possible inputs with more and more training data *k-Nearest Neighbor classifier*

### k-Nearest Neighbor classifier
  - Depends critically on having enough training data to adequitly fill in the input feature space.
  - "If there are three features, then the space is three-dimensional and the training data will fit inot some cube in that space"
  - **!** The classifier measures the distance between points in the **training data** and that of a new, unknown **feature vector** and votes on what **label** to assign
    - The dense the space is with training points, the more often the voting process will succeed.
    - Loosely speaking, data augmentation fills in the this space.
    - NN Could potentially learn the wrong thing

### Augmenting Training Data
 - We need to getnerat new samples from it that are `plausible`
  - images : straightforward, often `rotate` or `flip` or manipulate pixels or swap color bands
    - top to bottom flip probably wont be realistic
  - `feature vector augmentation` : more subtle, not always clear how to do it, or even plossible.
     - we can shift the color between red or green or blue
     - typically you try to augment continous values, crating a new feature vector that still represents the original class

### PCA (principal component analysis)
  - Old technique, used for over a **century**
  - Used to combat curse of dimensionality, by reducing the number of features in a dataset.
  - Example:
    Imagine we have a dataset with only 2 features. We plot on scatter plot and shift the origin to 0,0 by substracting the mean value of each feature, this in effect changes the origin. It tells you the direction of the variance of the data. The directions are the **principal components**
    **principal components** tell you how much of the variance of the dsta is explained by each of the directions.
    **How PCA can ehlp fight the curse** find the principa component and then throw the less influential ones away.
  - **why helpful** Once you know the principal components, you can use the PCA to create derived variables, which means you can rotate the data to align it with the principal components.
  - If we take the original data, x transform it to the new represation, and then the invers tranform it.
    - **modify** some of the principal components, we return a new set of samples that are not x but are based on x
  - **normally distributted** means that it follow the bell curve so that most of the time the value will be near the middle.

## Augmenting CIFAR-10
  - color images stored as RGB data
  - taken from the ground level, top down and bottom flips do not make sense. but left and right do.
  - Translations are a more common technique and small rotations
  - what to do with pixels that have no data after the shift or rotate?
    - options:
      - leave pixels black, or all 0 values. let model know there is no info there.
      - replace the pixels with mean value of the image, which we hope model will also ignore
      - Keras as tools for doing this via and **image generator object**
  - we dont want to hand random croppings of images but insted crop down to 28*28