## Objective
- Acquire raw data
- peprocess data (often most labor intensive)
- Create datasets used in rest of chapters

## resources
- [iris repository](https://archive.ics.uci.edu/ml/index.php)
- [iris download](https://archive.ics.uci.edu/ml/datasets/iris), [iris.data](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)
- [Wisconsin Diagnostic Breast Cancer dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data)
- [MNIST](yann.lecun.com/exdb/mnist)
- [CIFAR](https://www.cs.toronto.edu/%7Ekriz/cifar.html)

# Steps
1. Randomize the data
2. Replace class names with integer labels
3. 5-1.py

See page 86
1. Analyze the data, understand the structure
2. Standardize - features a re different scales 
3. Sort - hold some back and categorize.

## MNIST Data set
see : ch_5_03_randomize_mnist.py (pg 88)
- use keras, but (cononical source)[yann.lecun.com/exdb/mnist]
- create additional dataset from this initial Keras set
- Keras will return the datset as 3D Numpy Arrys.
  - First dimension is the number of images - 60,000 training
  - 10,000 for test.
  - the second and third dimensions are the pixels of images
  - the images are 28x28 in size. 8 bit integers [0, 255]

We will create additional datasets from the initial one.
  1. Unravel the images to form form feature vectors
  2. Permute the order of the images in the dataset. The reordering of the pixels will be deterministic and applied consitently across all images
  3. Create an unraveled feature vector version of the permuted images

## CIFAR-10 dataset
 - in Keras (pg 91)
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
- **Data augmentation** uses the data in teh existing dataset to generate new possible samples to add to the set. (92)
- We need Models to **interpolate** 
- **Data Augmentation** should be used wherever its feasible and small dataset
  - Take data we already have and modify it to create a new sample that could have plausibly come from the same parent distribution
- **Curse Of Dimensionality** (too many features) its solution is to fill in the space of possible inputs with more and more training data *k-Nearest Neighbor classifier*

### k-Nearest Neighbor classifier
  - It depends  on having enough training data to adequately fill in the input feature space. If there are three features the the space is 3 dimensional. The classifier measures the distance between points in the training data and that of a new, unknown feature vectore and votes on what label to assign.
  - The denser the space is with training points the more often the voting process will succeed.
  - Depends critically on having enough training data to adequitly fill in the input feature space.
  - "If there are three features, then the space is three-dimensional and the training data will fit inot some cube in that space"
  - **!** The classifier measures the distance between points in the **training data** and that of a new, unknown **feature vector** and votes on what **label** to assign
    - The dense the space is with training points, the more often the voting process will succeed.
    - Loosely speaking, data augmentation fills in the this space.
    - NN Could potentially learn the wrong thing

### Regurlization (94)
- When working with modern deep learning models we'll see that data augmentation has additional benefits.
- During training a neural network becomes conditioned to learn features of tehr training data. If the features are actually useful everything is ok. Sometimes the NN learns the wrong thing.
- Regularization helps the network learn important features of the training data.
- Data augmentation conditions the learning process to not pay attention to quirks of the samples but to instead focus on more general features.
- Data augmentation lessens likelihood of overfitting
- Be sure that augmented smple belongs to the same set.
- **The correct way to augment data is after the training, validation and test splits have been made.**

### Augmenting Training Data (94)
 - We need to generate new samples from it that are `plausible`
  - images : straightforward, often `rotate` or `flip` or manipulate pixels or swap color bands
    - top to bottom flip probably wont be realistic. D realistic modifications.
  - `feature vector augmentation` : more subtle, not always clear how to do it, or even plossible.
     - we can shift the color between red or green or blue
     - typically you try to augment continous values, crating a new feature vector that still represents the original class


### Augmenting Iris Dataset (SEE: ch_5_05_iris_data_set_PCA_plot.py) (PG 97)
- The iris dataset has 150 samples form 3 classes each with 4 features.
- We augment using PCA : principal comonent analysis, old method to combate curse of demantionality (pg 95)
- PCA tells you the direction sof the variance of the data. These directions are the **principal components**
- Principal components tell you how much of the variance of the data is explained by each of these directions.
- PCA is used to reduce the number of features while still, hopefully, representing the dataset well.

### PCA (principal component analysis) (PG 97)
- TLDR: Components in PCA are your primary features, we plot them in 2d space, so we normally are dealing with 2 features. WE shift the plot to the origin (0,0) by substracting the mean value of each feature. However we need to keep all the components for augmentation. So we list components in order of importantce. And we use this to augment. Knowing the principal components you can create derived variables. So you rotate the data to align it with the principal components. (use transform method). If we take original data, transform it modify some principal components, then inverse transform it we get data based on x.

`See ch_5_06_augementing_iris_with_pca.py`
1. Components are ordered in importance in pca.
2. We want to keep the most important components.
3. We dont want to transform too much.
4. We want to keep components that represent some 90% - 95% of the variance in the data.
5. Remaining components are modified adding normally distributed noise.
6. Normally distributted means it follows a bell curve, where most of the value is near the middle.

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

**important PCA notes**
This approach is appropriate for continuous features only. You should be careful to modify only the weakest of the principal components, and only a small amount.

**TRY** applying the same technique to augment the breast cancer dataset, which also consists of continous features.

## Augmenting CIFAR-10

For CIFAR-10 there are color images stored as RGB data. Translate - shiftting image in the x or y direction or both, small rotation are another technique.
For pixels that have no data after the shift or rotation. Simply leave the pixels black (all values 0), OR replace the pixels with the mean value of the image (which also provides no info)
We want the machine to disregard these.

The most popular solution is to crop the image.

Pulling  a random patch from the image of 28x28 pixes is the equivalent of shiffting

Rotating the image first requires interpolation of the pixels.
We take each 32x32 test input and crop it to 28*28 by droppin the outer 6 pixels. The center crop still represents the actual test image and not some augmented version

  - color images stored as RGB data
  - taken from the ground level, top down and bottom flips do not make sense. but left and right do.
  - Translations are a more common technique and small rotations
  - what to do with pixels that have no data after the shift or rotate?
    - options:
      - leave pixels black, or all 0 values. let model know there is no info there.
      - replace the pixels with mean value of the image, which we hope model will also ignore
      - Keras as tools for doing this via and **image generator object**
  - we dont want to hand random croppings of images but insted crop down to 28*28