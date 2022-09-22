# see https://youtu.be/lfqjQeKwNmI?t=218
from plotnine.data import diamonds

# 5k rows, each row is a diamond
print(diamonds)

# summary statistics
diamonds.describe()

# train_test_split() splits traiing/testing datasets
# KNeighborsClassifier() to fit to k-nearest model
# StandardScaler() to scale features used in models.
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# remove features 
y = diamonds['cut']
X = diamonds.drop(column=['cut', 'color', 'clarity'])

X_train, X_test, y_train, y_test = train_test_split(X, y)

knn = KNeighborsClassifier()

# fit
knn.fit(X_train, y_train)

# measure accuracy
print(knn.score(X_test, y_test)) #.55

# center + scale
ss = StandardScaler()
# do scaling process after you have split the data!
# fit = calculate mean and standard deviation of the features
# transform = substract the mean and divide by standard deviation
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.fit_transform(X_test)

knn.fit(X_train_scaled, X_test_scaled)

knn.score(X_test_scaled, y_test) # .70

# Other scalers in sklearn https://youtu.be/lfqjQeKwNmI?t=738