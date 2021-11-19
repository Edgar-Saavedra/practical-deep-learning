## Pre-reqs

### numpy
- python library to add array processing
### scikit-learn
- library containing traditional ML models
- uses NumPy arrays
- https://scikit-learn.org/stable/
### Keras with TensorFlow
- Many different kinds of CNN libraries
  - Keras
    - https://keras.io/
  - PyTorch
    - https://pytorch.org/
  - Caffe
    - https://caffe.berkeleyvision.org/
  - Caffe2
    - https://caffe2.ai/
  - Apache MXnet
    - https://mxnet.apache.org/versions/1.8.0/


### Instillation
```
# Install Python 3 that will be managed by Homebrew
brew install python3
# Get access to the scientific Python formulas
brew tap Homebrew/python
$ brew install scipy
$ brew install ipython
$ brew install jupyter
$ brew install numpy
$ brew install matplotlib
$ pip3 install scikit-learn
$ pip3 install pillow
$ pip3 install h5py
$ pip3 install keras
```

- pillow: image processing
- h5py : for working with HDF5 data files (scientific data)
- matplotlib: for plotting

`brew install python3; brew tap Homebrew/python; brew install scipy; brew install ipython; brew install jupyter; brew install numpy; brew install matplotlib ; pip3 install scikit-learn; pip3 install pillow; pip3 install h5py; pip3 install keras`

Machine learning is about building models that take input and arrive at a conclusion. The conclusion could be labeled to classify. The models learns on its own and learns by example. A model is y = f(x). Y is the continuous `class label` output. x is the `features` "unknown input" . Features are also measurements of the input that the model can use to learn what output to generate. f in y=f(x) is our algorithm or mapping for an unknow x and y. The machine is learning the parameters of the model. We are not cementing this algorithm, sometimes in NNs its hard to know the machine has learned.

## 3 Branches of manchine learning
Superviced learning, unsupervised learning, reinforcement learning.

### Supervised learning
We supervise the training of the model with a set of known x and y values, the `training set`. A dataset that is of known values is a `labeled dataset` since we know the y that goes with each x.

### Unsupervised learning
Trys to learn the parameters used by the model using only x

### Reinforcement learning
Train a model to perform a task. MOdel learns a set of actions to take given the state of its world. 

### Classical Models
- k-Nearest Neighbors
- Random Forests
- Support Vector Manchines



## Vectors
- (ROWS)
- `1 dimensional` list of numbers
- Used to respresent points in space
- In ML vectors are used to represent *INPUTS* to models

ROW VECTOR: `a = [0, 1, 2, 3, 4]`
Common Vector is vertical *COLUMN VECTOR*
```
[
  0
  1
  2
  3
  4
    ]
```

## Matrices
a 2D array of numbers `2 dimensional`
we index an entry by column and row
- (COLUMNS)
```
[ 1 2 3
  4 5 6
  7 8 9 ]
```

## Mulitplying Vectors x Matrices

- If a vaule is not found the multiplication  = undefined
- Many dimensions (3-4) we have a CNN also referred to as `tensor` a "stack" of matrices
- *mean* "artihmetic average We add all values together divide by # of values
- *standard of error of the mean*  This is the +/- qauntities from the mean. This comes along with the mean.
- *average spread* substract each value from the mean. Square each result. Add all the square values, divide by number of values we added minus 1.
- *variance* the number we get from the `average spread`.
- *standard deviation* take square root of `variance` 
- *median* the middle value, half of samples are above, half are below. 2 ways to get median: Sort the values numerically, find middle value. Great if we have odd number values, or  we take the  mean of 2 values. Median is sometimes better than the mean if the samples dont have a good even spread around the mean.
- *descriptive statistics* values derived from a dataset that can be used to understand the dataset.
- *probability distribution* the place where the numbers from the `descriptive statistics` come from aka *parent distribution* "the thing that generates the data we'll feed our model. The ideal set of data our data is approximating.
- 2 types of "paren distributions" *uniform* and *normal distributions*
- *uniform* where the results are representative of each sample equally. An oracle that will give us any of its allowed responses in a range in a equally weighted fashion.
- *normal distributions*  (Guassian Distribution). Visually a bell curve. Where 1 value is most likely. The likelihood of the other values decreases as one gets further from the most likely value. This most likely value is the `mean`. How fast our likelihood to get to non mean values goes to 0.
- *standard deviation* (noted as sigma) Is also the number that tells us how quickly the likelihood drops to zero (but never really reaches it)
- *statistical test* Measurement used to decide if a hypothesis is true or not. 
  We want to see if 2 sets of measurments are from the same parent distribution. If statistic  calculated is outside of a certain range we reject the hypothesis.
- *t test* (statistical test) Assumes our data is normally distributed. Since this assumption may or may not be true it is known as a "parametric test"
- *Mann - Whitney U test*  it helps us decide if two samples are from the same parent distribution but makes no assumptiona bout how the data values in the sample are distributed. It is known as `nonparametric test`
- *p value* value we get from parametric and nonparametric tests. If low (0.05 aka 1 in 20) we assume the hypothesis is not true. We now lowered the tresh hold 0.001 anything under that is "statistically significant"
- *GPU (graphics processing units)* Co-computers implemented on a graphics card. For video gaming. Used for NN models. NVIDIA the leader in creation of GPUs for deep learning via its Compute Unified Device Architecture (CUDA). There are different versions of Tensorflow to work with deep learning either on CPU or GPU.

### Multiplying Vectors
- need to know if either row or column vectors
`[1, 2, 3] * [4, 5, 6] = [4, 10, 18]`
 1 * 4 = 4
 2 * 5 = 5
 3 * 6 = 18

*Outer Product* 
On left. Outer product becomes a matrix
*Inner product*
On right (dot product), becomes a single number (scalar)
typically vector is on the right side of the matrix

- number of *columns* in matrix (left) matches number of elements in vector (column vector)
- out put is sum of products of the matrix

EXAMPLE: 

*Outer product (matrix)*
```
outerProduct = 
[ a b c
  d e f ]
```
*Inner product (vector)*
```
innerProduct
[
  x
  y
  z
] 
```
*Scalar*
```
[ 
  ax + by + cz
  dx + ey + fz
]
```

EXAMPLE: 

*Outer product (matrix)*
```
outerProduct = 
[ 
  a b c
  d e f 
]
```
*Inner product (vector)*
```
innerProduct =
[
  A B
  C D
  E F
] 
```
*Scalar*
```
[ 
  aA + bC + cE    aB + bD + cF
  dA + eC + fE    dB + eD + fF
]
```
