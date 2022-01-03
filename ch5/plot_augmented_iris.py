import numpy as np
import matplotlib.pylab as plt

originalFeatures = np.load('/Users/esaavedra/Work/personal/projects/practical-deep-learning/ch5/iris_features.npy')
augmentedTrain = np.load('/Users/esaavedra/Work/personal/projects/practical-deep-learning/ch5/iris_train_features_augmented.npy')
augmentedTest = np.load('/Users/esaavedra/Work/personal/projects/practical-deep-learning/ch5/iris_test_features_augmented.npy')

print('originalFeatures', originalFeatures)
print('augmentedTrain', augmentedTrain)

plt.scatter(originalFeatures[:, 0], originalFeatures[:, 1], marker='o', color='pink')
plt.scatter(augmentedTest[:, 0], augmentedTest[:, 1], marker='o', color='blue', alpha=0.5)
plt.scatter(augmentedTrain[:, 0], augmentedTrain[:, 1], marker='.', color='black', edgecolors='none', alpha=0.5)
# ax = plt.axes()
plt.show()