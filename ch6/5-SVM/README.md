# Support Vector Machines 125

- margin
- support vectors
- optomization
- kernels

A classfier can be thought of as locating one or more planes that split the space of the training data into homogeneous groups.

How do we use what plane (line) to use.

Maximal margin - location were the margin is defined as the distance form the closest sample points.

The goal of an svm is to lacte the maximum margin position, 

### Support vectors
Vectors that support the magin. They define the margin. 

### optimization

To find the maximm margin hyperplane by solving and optimization problem. 

We have a quantity that depnds on certain parameters and we want to find the set of paremeter values that makes the quantity as small or large as possible.


### Orientation

We have a vector and offset ( which we mush find ). 

### Quadratic programming
see pg 127


### Fudge factor 
The variable that affects the size of the margin found.
This is the hyperparameter of the SVM a value tat we need to set to get the SVM to train properly.


### Kernels
SVM uses inner products. We transpose a vector and apply matrix multiplication.

The kernel is this prodcut multiplication.

Linear kernel - vector transposed (row)

Other kernels are possible

#### Guassian Kernel (RBF)
Radial basis function Kernel. 

This kernel introduces a new parameter - it relates to how spread out the Gaussian Kernel is around a praticular training point.

One uses grid search over C and if using the RBF Kernel parameter, to locate the best perfoming model.

Support Vector Machines use training data mpped through a kernel function, to optimize the orientation and location of a hyperplane, that produces maximum margin between the hyper plane and the support vectors of the training data. The user needs to select the kernel function and associated parameters so that the model best fits the training data.