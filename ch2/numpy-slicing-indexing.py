# https://numpy.org/doc/stable/reference/arrays.indexing.html
# NumPy’s basic slicing is an extension of Python’s basic slicing concept extended to N dimensions.
# allows you to take slices of an array along its dimensions using basic slicing notation, i.e start:stop:step.

a = array([['A', 'B', 'C', 'D', 'E'],
       ['F', 'G', 'H', 'I', 'J'],
       ['K', 'L', 'M', 'N', 'O'],
       ['P', 'Q', 'R', 'S', 'T'],
       ['U', 'V', 'W', 'X', 'Y']])

# So far so good. We’ve basically sliced the array along the first axis to get up to the second row starting at index 0 
# (note that the stop index is not included!), and along the second to get columns one to three.
# Think of the axes just as the (x,y,z) dimensions of a matrix. 
# Where a third dimension can be simply though of as stacking multiple 2D arrays together:
a[:2, 1:4]
# array([['B', 'C', 'D'],
#        ['G', 'H', 'I']])

# start:stop:step.
# row,column,level

# Advanced indexing
a_indexed = a[[4,3,1], [2,4,0]]
# array(['W', 'T', 'F'])

# advanced indexing follows a different set of rules
# basic slicing, we are indexing on a grid which is defined by the slices we take on each dimension
# advanced indexing can be thought of as specifying a set of (x,y) coordinates of the values we want to retrieve.
# [4,3,1],[2,4,0] , will behave as indexing on(4,2) , (3,4) and (1,0) respectively. 
# Which basically means that we will retrieve as many elements from a given axis as indexes specified in the indexing array into that dimension.


# broadcasting
#  how NumPy treats arrays with different shapes during arithmetic operations. Subject to certain constraints, 
# the smaller array is “broadcast” across the larger array so that they have compatible shapes

#  It is stated that two dimensions are compatible when

#     they are equal, or
#     one of them is 1

# NumPy will try to make the shapes from the indexing arrays compatible before performing the indexing operation.
rows = np.array([0,2,1])
cols = np.array([2])
ix = np.broadcast(rows, cols)
print(ix.shape)
# (3,)

print(*ix)
# (0, 2), (2, 2), (1, 2)

a[rows, cols]
# array(['C', 'M', 'H'])

rows = np.array([4,3,1])
cols = np.array([2,4,0])
# So we have to find a way for NumPy to understand that we want to retrieve a grid containing all values.
# The answer is… to USE BROADCASTING!!
# What we can do is to add an axis to one of the arrays,

rows[:,np.newaxis]
array([[4],
       [3],
       [1]])
# Or we could equivalently use rows[:,None]

ix = np.broadcast(rows[:,None], cols)
print(*ix)
# (4, 2) (4, 4) (4, 0) (3, 2) (3, 4) (3, 0) (1, 2) (1, 4) (1, 0)

# This broadcasting can also be achieved using the function np.ix_:
a[np.ix_(rows, cols)]
# array([['W', 'Y', 'U'],
#        ['R', 'T', 'P'],
#        ['H', 'J', 'F']])

# Example: And we want to take these columns respectively along the first axis, 
# so columns 3 and 4 from the first row, 0 and 2 from the second row and so on. How can we do that?
cols = [[3,4], [0,2], [0,1], [1,2], [3,3]]
rows = np.arange(a.shape[0])
# this causes error
# a[rows, cols]
# we must add a new axis to rows
a[rows[:,None], cols]

# Combining advanced indexing and basic slicing

a = np.arange(60).reshape(3,4,5)
print(a)
# array([[[ 0,  1,  2,  3,  4],
#         [ 5,  6,  7,  8,  9],
#         [10, 11, 12, 13, 14],
#         [15, 16, 17, 18, 19]],

#        [[20, 21, 22, 23, 24],
#         [25, 26, 27, 28, 29],
#         [30, 31, 32, 33, 34],
#         [35, 36, 37, 38, 39]],

#        [[40, 41, 42, 43, 44],
#         [45, 46, 47, 48, 49],
#         [50, 51, 52, 53, 54],
#         [55, 56, 57, 58, 59]]])


# One-dimensional slices
# The general syntax for a slice is array[start:stop:step]. Any or all of the values start, stop, and step may be left out (and if step is left out the colon in front of it may also be left out):

# A[5:]
# array([5, 6, 7, 8, 9])
# A[:5]
# array([0, 1, 2, 3, 4])
# A[::2]
# array([0, 2, 4, 6, 8])
# A[1::2]
# array([1, 3, 5, 7, 9])
# A[1:8:2]
# array([1, 3, 5, 7])