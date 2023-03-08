pg 112
# k-Nearest Neighbor

Instead of computing per class centroid we use the training data as is and selected the class label for a new input sample by finding the closest member the trainig set and using its label.

Example: If we are using one neighbor we call the classifier a 1-Nearest Neighbor or 1-NN Classifier

## Example

Lets assume data set has spread out samples.
We have 2 features and 4 classes with 10 examples and assume k=3. We look at the 3 closest data points closets to our sample. By majority we select the class that has the most neighbors. However when there is an equal spread of closest neighbor classes the simplest choice is to select the class label at random.

In KNN the model is the data the Model and no training is necessary.
Curse of dimensionality is still an issue unless if feature space is small.