# PAGE 84
import numpy as np

# load text file
with open("iris.data") as f:
    lines = [i[:-1] for i in f.readlines()]

# Create vector labels by converting the text label into integer 0-2
n = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
# print(lines[0])
# each line is comprise of a feature that looks like: 5.1,3.5,1.4,0.2,Iris-setosa
# we get the last item and use that as the label and use the index from 
# the previous array in place of label
x = [n.index(i.split(",")[-1]) for i in lines if i != ""]

# LABELS
# A list list of Labels
# turn the list into a numpy array
x = np.array(x, dtype="uint8")


# ```
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

# ```
# https://scipy-cookbook.readthedocs.io/items/Indexing.html

# FEATURES
# take the numbers from each line and append to array in new row
# then convert to nump
y = [[float(j) for j in i.split(",")[:-1]] for i in lines if i != ""]
y = np.array(y)

# ndarray.shape
# Tuple of array dimensions.

# The shape property is usually used to get the current shape of an array, but may also be used to reshape the array in-place by assigning a tuple of array dimensions to it.
# print(np.random.random(x.shape[0]))

# argsort returns the indecies that sort the array
i = np.argsort(np.random.random(x.shape[0]))


x = x[i]
y = y[i]

np.save("iris_features.npy", y)
np.save("iris_labels.npy", x)
