## Classical machine learning

### 6 Classical models
Nearest Centroid, k-Nearest Neighbors, Naïve Bayes, Decision Trees, Random Forests, Support Vectors Machines

We assume we mave m samples of each of the n classes, we assume our dataset is properly designed and no validation samples needed.
Our goal is to have a model that uses the training set to learn. So that we can apply the model to the test set.

If features are well chose we might expet a plot of the points in the w-dimensal space to group the classes.

Goal is to assign a class label to the new point either square, star, circle or tirangle

#### Nearest Centroid aka (template matching)
- We can think that we could represent each group (class) by an average position in the feature space
- the average point of a group of points has a name **the centroid** aka "center point"
  - **to compute** : add them up and divide by how many we have.
  - we first fifnd the average of all the x-axis and then the average of all the y-axis. If we have w-dimensions, we'll do it of each of the dimensions
  - in the end we end up with a single point
  #### Important
  - **Euclidean distance** How is it helpful? I will be a point in the feature space. We can then measure the distance between this poin and each of the centroids and assign the class label of the closests centroid
  - a large feature space implies that this might not be the right approach

### k-Nearest Neighbors
Instead of computing per class centroids.Use the training data as is and select the class label by finding the closest memeber of the training set and using its label.
- We can look a "K" number of neighbors
- we select the class label at random, since any might be likely
- no training step is necessary
- dimensionality is an issue
- there needs to be a balance between training data size, which leads to better performance
- better at diffusing overlapping class groups

### Naïve Bayes
- **posterior probability** represents probability atht event A has occured given event B has already occured
- If we know the likelihood of having x be our feature vector given that y is the class, and we know how often class y shows up then we can clculate the probability that the class of the feature x is y.
- We can calculate by making a histogram of how often each class label shows up in the training set.
- We assume that each of the features in x is statistically independent, this assump tion often enough true that we can get by.
- When 2 events are independent, their joint probability is simply the product of their indivual probabilities
- If we know the probability of measuring a particular value of a feature, we can get the likelihood of the entire feature vector given the class lable by mulitiplying each of the per feature probabilities together.
- **mean value** is the the average value we'd expect if we drew samples from the distribution repeatedly.
- **standard deviation** (spread) a measure of how wide the distribution is (how spread out it is around the mean)
- **some math** The likelihood of a particular feature value, given the class is y, is distributed around the mean value we measured from the training data according to the normal distribution. Then multiply the resulting probabilites together and then multiply that value byt the prior probability of class 0 happening ***Laplace smoothing*** we claim tha a  "good"training set will represent all possible values for the features.

### Decision Trees
- A decision tree is a set of nodes. The nodes either define a condition and branch based on the truth or falsehood of the condition. Nodes that do not branch are called **leaf nodes**
- Decision trees are called trees because they branch. **DecisionTreeClassifier**.
- The first node in the tree is the **root**. 
- Decision trees are handy in situations where the "why" of the class assignment is as important to know as the class assignment.
- The algorithm to build a Decision Tree is recursive.
  - It starts at the root node. Determines the rule for the node and then calls it self on the left and right branches. The stopping conditino is the leaf node. The recursion terminates, it returns the leaf's parent and calls itself on the right branch and so on..
- Decision trees are greedy algorithms, every node it selects the best rule for the current set of infromation available to it. It runs through all possible compbinations of features and values making continuous values discrete by binning and evalutates the purity **Gini Index** (we seek to minimize the Gini Index) of the left and righ training sets after the rul is applied. The best-performing rule is the on kept a that node.
- Decision Trees are useful when the data is discrete or categorical or has missing values.
- They have a tendency of **overfitting** - learning meaningless nuances.
- They grow very large as the number of features grows.

### Random Forest
- Overfitting can be mittigated
- uses:
  - ensembles of classifiers and voting between them
  - resamples the training set by selecting sample with replacement
  - selection of random feature subsets
- **bootstrap sample**/**bagging**: selecting a new training set from the original training set. models built from the collection fo resampled datasets that build the Random Forest. This creates a fores of  "trees". We randomly select a subset of features from the tree to increase the overall robustness of the forest.

### Support Vector Machines
Uses the training data mapped through a kernel function to optimize the orientation and location of a hyperplane that produces the max margin between hyperplane that produces the max magin.
- Key concepts **margins**, **support vectors**, **optimization**, **kernels**
- Margins: The goal of the classifier is to separate the training feature space into groups using hyperplanes. The classifier will locate on or more "planes" that split the space of training data into homogeneous groups. The margin is a distance from the closest sample point. The goal is to locate the max margin position where we can be most certain to not misclassify new samples, given the knowledge gained from the training set.
- **support vectors** the support vectors which help define the margin.
- **optimization** A way to find the max hyperplane to solving an optimization problem via a technique called **quadratic programming**. It includes a fudge facto C which affects the size of the margin found. C is a hyperparameter we set.
- **Kernels** , (Guassian Kernel, radial basis function), it relates to how spread out the Gaussian kernel is around a particular training point.