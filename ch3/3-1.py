# This file shows the usage of numpy arrays. We do things such as:
# - Example 1: Create Lists from a range and random numbers
# - Example 2: Time how long it takes to create a list of random list from example 1
# - Example 3: Use the append method
# - Example 4: Pre allocate array with slots and populate them
# - Example 5: Perform numpy array arithmatic to create array from 2 other arrays.

# Create 1M elments mulitpy the two lists as quickly as possible
# This experiment shows that using Numpy is 25X faster than pure python with naive implementation.

# import numpy
import numpy as np
import time
import random
n = 1000000
# Example 1 : Create 2 lists
a = [random.random() for i in range(n)]
b = [random.random() for i in range(n)]

# Example 2
s = time.time()
c = [a[i] * b[i] for i in range(n)]
print(f"copmrehension: {time.time()-s}")

# - Example 3
s = time.time()
c = []
for i in range(n):
  c.append(a[i] * b[i])
print(f"for loop: {time.time() - s}")

# - Example 4
s = time.time()
c = [0] * n
for i in range(n):
  c[i] = a[i] * b[i]
print(f"existing list: {time.time() - s}")

# - Example 5
x = np.array(a)
y = np.array(b)
s = time.time()
c = x * y
print(f"Numpy time {time.time() - s}")
