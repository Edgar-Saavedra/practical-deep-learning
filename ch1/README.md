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

## Vectors
- 1 dimensional list of numbers
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
a 2D array of numbers
we index an entry by column and row
```
[ 1 2 3
  4 5 6
  7 8 9 ]
```

## Mulitplying Vectors x Matrices

### Multiplying Vectors
- need to know if either row or column vectors
`[1, 2, 3] * [4, 5, 6] = [4, 10, 18]`

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