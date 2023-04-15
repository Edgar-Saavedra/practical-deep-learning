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
- Normalized means that per feature, each value has the mean for that feature substracted and then is divided by the standard deviation.
- Normalization maps all the features into the same overall range so that the value of one feature is similar to the value of another.

## Classical Model Summary

### Nearest Centroid
Simplest Model, not really used, unless task is easy.

You could use a more general approach that first finds and appropriate number of centroids and then groups them together to build the classifier.

- Pros
  - Takes only a handful of code.
  - Not restricted to binary models.
  - Trains very fast
  - Memory overhead is small: only one centroid is stored per class
  - All that needs to be computed is distance from the sample to each class centroid
- Cons
 - Makes simplistice assumption about the distribution of the classes in the feature space.
 - only accurate if classes form a tight group and the feature space and groups are distant from one another.

### K-Nearest

Simple model to train there is no training: we store the training set and ust it to classify new instances by finding `k` nearest training set vectors and voting.

- Pros
  - no trainig required
  - can perfom well if trinig samples is large relative to feature space.
  - multiclass support is implicit.
- Cons
  - Classsification is slow - needs to look at every exmple to find the nearest neighbors

### Naive Bayes
Simple and efficient, and valid even when core assumption of feature set isn't met.

- Pros
  - Fast to train
  - Fast to classify
  - Support multi class
  - So long as the probability of a particular feature value can be computed

- Cons
  - Feature independece assumption is seldome true in practice.
  - The more correlated the features the poorer the performance.
  - Works with discrete features, using continuous features often involves a second leve of assumption. We need to estimate the paremeters of the distribution from the dataset instead of using histograms to stand in for the actual feature probabilities.

### Decision Trees
Useful when its important to be able to understnad in human terms why a class was selected.

- Pros
  - Fast
  - Fast to use for classifying
  - Supports Multiclass
  - Not restricted to continuous features.
  - Can justify its answer by showing series of questions aske from root.
- Cons
  - Prone to overfitting 
  - Interpretability degrades as the tree size increases.
  - Tree depth needs to be balanced wiht quality of decisions which affects error rate.

### Random Forest
More powerful form of Decision Trees. Uses randomness to reduce the overfitting problem.

- Pros
 - Support multiclass
 - Resonably fast to train
 - Accuracy imporoves with diminshing returns
- Cons
  - Interapretability disapeares. Combined effect of forest as a whole can be difficult to understand.
  - Runtime scales linear with the number of trees. (can be mitigated with parallization)

### Support Vector Machines
Before rebirth of Neural Networks. 

- Pros
  - can show excellent performance when tuned properly.
- Cons
  - Multiclass models are not supported.
  - Expect only continuous features.
  - normalization and other scaling is necessar
  - large datasets are difficult ot train (when using other than linear kernels)
  - require careful tuning of margin and kernel parameters (C, gamma), can be mitigated by search algorithms that seek the best hyperparameters

## When To Consider a classical model.

- When the dataset is small (tens or hundreds of examples)
- WHen computational requirements must be kept at minimum (hanheld devices, microcontroller) : Neares Centroid, Naive Bayes. So are Decision Trees and Support Vector Machines once trained.
- Some models can explain themselves (Decision Trees, K-NN, Neares Centroid, Naive Bayes) 166
- If the input to the model is a vector (NOT MULTI DIMENSIONAL FEATURES) 
- No need to look for structure among the features
- When the input is a feature vector without spatial structure specially if features are not related to one another.
