import numpy as np
from matplotlib import pyplot as plt

train_data = np.load("iris2_train.npy")
train_labels = np.load("iris2_train_labels.npy")

test_data = np.load("iris2_test.npy")
test_labels = np.load("iris2_test_labels.npy")

print(train_data[:, 0], train_labels.shape)
print(train_data[:, 1].shape, train_labels.shape)

plt.scatter(train_labels, train_data[:, 0])
# plt.scatter(train_data[:, 1], train_labels)
# plt.scatter(test_data[:, 0], test_labels)
# plt.scatter(test_data[:, 1], test_labels)
plt.xlabel("$x_0$", fontsize=16)
plt.ylabel("$x_1$", fontsize=16)
plt.show()