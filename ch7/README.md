## Using sklearn to create classical models

To build intuition about how the different models perform relative to one another.

Using three datasets
- iris dataset (original, augmented)
- breast cancer dataset
- Vectorized MNIST dataset

## Iris dataset
- Has 4 continuous features
  - sepal length, sepal width, petal length, petal width
- 150 samples, 50 eachf from 3 classes
- WE applied PCA augmentation (pg 95)
- We will use sklearn to implement 
  1. Nearest Centroid
  2. k-NN
  3. Naive Bayes
  4. Decision Tree
  5. Random Forest
  6. SVM


## Breast Cancer Dataset
- We will use the normalized dataset
- Normalized means that per feature, each value has the mean for that featur substracted and then is divided by the standard deviation.
- Normalization mpas all the features into the smae overall range so that the value of one feature is similar to the value of another.

