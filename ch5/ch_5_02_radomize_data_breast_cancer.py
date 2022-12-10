import numpy as np
import matplotlib.pyplot as plt

# Read text in raw data
with open("wdbc.data") as f:
    lines = [i[:-1] for i in f.readlines() if i != ""]

# Extract each label
n = ["B", "M"]
x = np.array([n.index(i.split(",")[1]) for i in lines], dtype="uint8")
# extract features
y = np.array([[float(j) for j in i.split(",")[2:]] for i in lines])

# randomize
i = np.argsort(np.random.random(x.shape[0]))
x = x[i]
y = y[i]

# - Calculating Standardization
#     - `x = (x-x.mean(axis=0)) / x.std(axis=0)`
#   - One must apply Standardization Or Normalization to most datasets. We want a mean of 0 and standard Deviation of 1.
#   - One must apply Standardization Or Normalization on the following:
z = (y - y.mean(axis=0) / y.std(axis=0))

np.save("bc_features.npy", y)
np.save("bc_features_standard.npy", z)

np.save("bc_labels.npy", x)
plt.boxplot(z)
plt.show()
