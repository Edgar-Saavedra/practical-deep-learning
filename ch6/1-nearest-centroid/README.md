pg 109
# Neares Centroid

- Take a dataset of N Classes with M samples for each
- There are training and test samples
- No need for validation samples
- Goal: take training samples to learn so we can apply the te model to the test set
- We want a vector space and components that distinct the classes
- We have w features
- We plot in a w-dimensional space
- Our plot is nicely seperated by group


### Centroid

The centroid is the average point of a group of points. 

In a 2-D space we find the average of all the x-axis coordinates and then the average of all the y-axis
In teh end we end up with a sinble point that we can use to represent the entire group.

** We can measure the distance between this point and each of the centriods and assign the class label of the closes centroid.


### Distance

Euclidean distance -  a straight line between two points. (pg 110)

### Nearest Centroid Classifier

Sometimes called template matching. The training data is used as a proxy for the class as a whole. New samples use those centroids to decide on a label.

### Flaws

The curse of dimensionality. As the number of features increases the space gets larger and larger and we need more data to know what the centroid should be. Classes need to be easily divisible. 
