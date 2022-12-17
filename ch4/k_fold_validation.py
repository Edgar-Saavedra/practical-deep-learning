# https://www.youtube.com/watch?v=eTkAJQLQMgw
# we want to predict incomig data
# we want to make sure our model doesn't over fit
# we split data into training and test set
# certain algorithms will have hyper params
# bias : we see how well a model fits data by bias - high bias good
# variance : how well it does on a new data set
# ridge regression: will have hyperparameter alpha, will prevent model fitting well to training data and will have a better variance

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

housing = fetch_california_housing()

housing_features = pd.DataFrame(housing.data, columns=housing.feature_names)

X = housing_features['RM'].values.reshape(-1, 1)

y = housing.target

plt.scatter(X, y)
plt.title('housing house prices')
plt.xlabel('average number of rooms per dwelling')
plt.ylabel('house prices')
# plt.show()

train_X, test_X,trin_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
alphas=[1, 1e3, 1e6]
regressor = RidgeCV(alphas=alphas, store_cv_values=true)
regressor.fit(train_X, train_y)
cv_mse = np.mean(regressor.cv_values_, axis=0)
print(alphas)
print(cv_mse)
print(regressor.alpha_)

# cross validation is better sometimes becuase it uses less data
# k-fold split up our data into multiple folds

predict_y = regressor.predict(test_X)
plt.scatter(test_X, test_y)
plt.plot(test_X, predict_y, color='red')
plt.title('housing house prices')
plt.xlabel('average number of rooms per dwelling')
plt.ylabel('house prices')
plt.show()