# This file shows how numpy arrays can be crated and manipulated.
# example using the Numpy.array() method
import numpy as np

# create basic numpy array
a = np.array([1,2,3,4])
print(f"array {a}", f"array size : {a.size}", f"array shape : {a.shape}", f"array dtype :{a.dtype}")

# one can imply the data type being used
# https://numpy.org/doc/stable/reference/arrays.ndarray.html#array-attributes
b = np.array([1,2,3,4], dtype="uint8")
print(b.dtype)

c = np.array([1,2,3,4], dtype="float64")
print(c.dtype)
# https://numpy.org/doc/stable/reference/arrays.dtypes.html#arrays-dtypes

# 2D array
d = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(d.shape, d.size, d)

# 3D array
d = np.array([
  [
    [1,11,111],[2,22,222]
  ],
  [
    [3,33,333],[4,44,444]
  ]
])
print(d.shape, d.size, d)

# defining arrays with 0s and 1s
x = np.zeros((2,3,4))
print(x.shape, x.dtype, x)
b = np.ones((10,10), dtype="uint32")
print(b.shape, b.dtype, b)

# multiply by number and numpy.ones()
y = 10 * np.ones((3,3))
print(y.shape, y.dtype, y)
# astype() - returns a copy of the array casint each element to the given data type
print(y.astype("uint8"))

# indexing
b = np.zeros((3,4), dtype="uint8")
b[0,1] = 1
b[1,0] = 2
print(b)

# range + slicing
a = np.arange(10) 
print(a, a[1:4], a[:6], a[6:])
# get the last element
print(a[-1])
# return vector in reverse order
print(a[::-1])

# creating matrix
b = np.arange(20).reshape((4,5))
print(f"creating matrix {b}")

# shorthand slicing
c = np.arange(27).reshape((3,3,3))
a = np.ones((3,3))
print(f"c : {c}")
print(f"a : {a}")

# replace second row in matrix
c[1,:,:] = a
print(f"c : {c}")
# or this is the same
c[0,...] = a
print(f"c 2 : {c}")

#broadcasting
a = np.arange(5)
# reverse
c = np.arange(5)[::-1]
print(f"a : {a*3.14}")
print(f"a : {a*a}")
print(f"a : {a*c}")
# // is integer division
print(f"a : {a//(c+1)}")

# matrix math operators
a = np.arange(9).reshape((3,3))
b = np.arange(9).reshape((3,3))
print(np.dot(a, b))