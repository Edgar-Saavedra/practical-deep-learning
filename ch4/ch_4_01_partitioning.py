import numpy as np
from sklearn.datasets import make_classification

# generate dummy data set
# Take 2 classes with 20 features to generate 10,000 samples
# 90 percent of the samples are in class 0 and 10 percent are in class 1
# our split ideally represents the parent distribution
x,y = make_classification(n_samples=10000, weights=(0.9, 0.1))
print(x.shape)

print(len(np.where(y == 0)[0]))
print(len(np.where(y == 1)[0]))