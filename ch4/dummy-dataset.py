import numpy as np
from sklearn.datasets import make_classification
# Use two classes and 20 features to generate 10,000 samples
# Samples x and Labels y
# make_classification accepts the number of samples requrested and the fraction for each class
x,y = make_classification(n_samples=10000, weights=(0.9, 0.1))
print(x.shape)
# np.where calls all the class 0 and class 1 instances
len(np.where(y == 0)[0])
len(np.where(y == 1)[0])
