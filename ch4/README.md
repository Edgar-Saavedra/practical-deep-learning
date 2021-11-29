## Working with data

- Dataset represents the data the model will encounter in the wild.
- **Classes** : models that put things into descrete categories
- **Label** : Identifier for each input in our **training set**. Example: a string or a number like 0/1. Models dont know what their inputs represent. Class labels are usually integers starting with 0. The ouput for our model.
- **features** inpputs for our model. Usually numbers. Numbers we want to use as input. The trainig of model trys to learn relationship between the input features and the outbut label. 
  - Model tryos to take feature **vector** with unknown labels and it trys to predict label
  - If model makes repeated predictions, a posibility is the selected featurs are not sufficiently capturing the relationship.
  - often need to be manipulated before they can go into model.
  - Can be floating or interval numbers
- **numbers**
  - Floating point : continuous infinite, between and intger and next
  - interval value: dsicrete, leave gaps in between. Linear.
  - ordinal: express an ordering
  - categorical: numbers as codes. **note** machine learning expect at least ordinal. We can make categorical at leas ordinal. We pay a price using categorical, they must be mutually exclusive only 1 in each row.

## Feature selection
- Feature vectors should contain only features that capture aspects of the data, that allow the model to generalize to new data.
- Should capture aspects of the data taht help the model seperate the classes.
- we need enough feature to capture all the relevant parts.
- too many features we fall victim to **cures of dimensionality**
- **Interval notation** `[` include value `(` exclude value.
- usually 2-3 dimensional vectors.
- As number of features increase the number of data needed needs to increase.

## A Good Datase
- In supervised learning we are the teacher.
- **Interpolation** estimating within a certain known range.
- **Extrapolation** Use the data we have to estimate outside the known range. Beyon what is known.
- Models more accurate when we interolate.
- **Linear Regression** The best fitting line to plot through data.
- We need comprahensive training data. Akin to interpolation.
- Dataset must cover full range of variation, within classes the model with see.