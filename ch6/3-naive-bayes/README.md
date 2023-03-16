## Naive Bayes

Used in natural language processing research

- Uses Bayes theorem
- Gives us the probability of something happening (A) given that we already know something else has happened (B) 

We want to know whether a feature vector belongs to a given class. 

If know the likelihood of having our feature vecotre given a class and we know how often the class shows up then we can calculate the probability that the class of the feature vector 

We can select the highest probability and label the input feature vector as belonging to that class.

We are trying to get P(y) - prior probability of the class

We assume that each of the features in the vector are statistically independent. 

Its naive to assume he features are independent

Joint probability- when two events are independent their joint probability that both happen is simply the product of their individual probabilities.

We build a historgram of each feature vector

1 We build a table of probabilities for each feature vector on each class.
2 We multiply each of those probabilieties
3 Multiply by the prior probability

P(xi|y)

We assume that features all follow normal distributions.
A normal distribution is defined by its mean and standard deviation

We can approximate mean and SD from the training data.

Given a new sample - the likelihood of a particular feature value, is distributed around the mean value we measure from the training data according to the normal distribution. This assumption is made on top of independence of each feature.

We then multiply the probabilities together. And multiply that by the prior probability of class 0 happening, we repeat this for each class. We end up with m numbers of the probabilities of the vectors belonging to each class. We select the largest probability and label as being the class.

Laplace smoothing - addresses for uncommon instances.

MultinomialNB in sklearn

https://www.youtube.com/watch?v=O2L2Uv9pdDA