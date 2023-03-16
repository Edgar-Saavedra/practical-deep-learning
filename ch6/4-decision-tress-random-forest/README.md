# Decision Tree

A decision Tree is a set of nodes. The nodes define a conditino and branch based on the truth or falsehood of the condition, Nodes that don't branch are called leaf nodes. 

Decision trees are called trees because they branch like trees.

sklearn DecisionTreeClassifier

The root - the first node at the top of the tree. Fore any new feature vector the series of questions is asked starting with the root node and the tree is travesers until a leaf node is reached to decide the class label.

## Great for
Decision Trees are useful when the data is discrete or categorical or has missing values.

## Recursion

The algorithm starts with troot node, determins the proper rule of that node and then calls itself on the left and right branches. The call on either branch will start again as if it were the root node. This will continue until a stopping condition is met.

The stopping condition is the leaf node. Once a leaf node is created, the recursion terminates and the algorithm returns to that leafs parent node and calls itself on the right branch.

Once both recursive calls terminate and subtrees are created, the algorithm returns to that node's parent and son on until the entire tree is constructed.

## Training Data

The trainnig data is presented as n samples. The set of samples used to pic the rule the root node implements. Once that rul has been selected we have two new sets of samples, one on left ans right.

The recursion then works with these nodes, using their respective set of training samples.

The Decision Tree algorithm is greedy, it uses brute force to locate candidate rules. I runs through all posible combination of freatures and values making continuous values discrete by bining, best performing rule is the one kept for each node.

## Purity
Best perfoming is determined by the purity of the branches. This determined by the Gini index.The algorithm seeks to reduce the Gini index by selecting the candidate rule that results in the smallest Gini.

# Random Forests

Decision Trees have a habit of overfitting the training data. Likely to learn meaningless staitstical nuances. Decision trees grow large as features grow.

Overfitting can be mittigated by Random Forests. You probably want to consider using Random Forests firsts.

- Ensembles of classifires and voting between them
- Resampling of the training set by selecting samples with replacement
- Selection of random feature subsets.

## Ensemble
We have a set of classifiers each trained on different data, like k-NN or Naive Bayes, we can use their outputs to vote on the actual category to assign to a known sample. 

Imagine an ensemble or forest of Decision Trees, but unless we do something more with the training set we'll see the same tree.

## Selection with replacement
Select a new training set from the original training set but allow the same training set sample to be selected more than once.

## Bootstrap sample 
A data selected in the `Selection with replacement` manner

## Bagging
Building a collection of new datasets from Boostrap Samples. Its model build from this collection of resampled datasets that build Random Forests.

If we retrain multiple trees with resampled training set with replacement, we'll get a forest of trees, each one slight slightly different from the others. This along with ensemble voting will improve things.

Note: if some of the the features are highly predictive they will dominate.

## Randomness
Wat if we we also randomly selected fore each tree in the forest a subset of the features. Will break the correlation between the tress and increase the overall robustness of the forest. If there are n features per feture vector each tree will select randomly squareroot n of them over which to build the tree.
