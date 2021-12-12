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

## A Good Dataset
- In supervised learning we are the teacher.
- **Interpolation** estimating within a certain known range.
- **Extrapolation** Use the data we have to estimate outside the known range. Beyon what is known.
- Models more accurate when we interolate.
- **Linear Regression** The best fitting line to plot through data.
- We need comprahensive training data. Akin to interpolation.
- Dataset must cover full range of variation, within classes the model with see.
- **classification** requires a comprehensive training data. The data must cover full range of variations withi the classes the model will see.
- **Parent Distribution** The ideal data generator. Training, test and data we give to model all come from parent distribution.
- **uniform parent distribution** When each value is equaly likely to happen.
- **prior class probability** probability with which each class in in the dataset in the wild. We generally want our dataset to match this.
  - Sometime hard to match this. We might start with an even number class instances and then change to a mix that matches the **prior class probability**
- **confusers** hard negatives to allow more precise features of a class. Example: having a not a dog category.
- **Capacity** & **complexity** **Capacity** : Number of parameters **complexity** How many parameters it can support relative to amount of training data. Usually good to have more training examples than model parameters. But Neural Networks  can work when there is less train data than parameters.
  - When the task isnt so complex we can get away with fewer training examples. But need more training data when more complex
  - **Get as much as practical**

## Data Preparation (Features)
- **Scaling** Is when we make every feature continuous.
  - Having a balance data set with representation of the parent distribution is optimal
  - Scaling features allows you to makeranges in numbers more similar.
  - **Mean Centering** Subtract the mean (average) value of the features over the entire dataset (sum each value divided by the number of values). We are shifiting data down toward 0.
    - **For images** mean centering is often done by substracting a mean image from each input image *Look more into this*
  - **Changing Standard deviation to 1** To spread the values around the mean. AKA **standardization/normalizing**
    - Whenever possible **standardize dataset** so that the features have 0 mean and standard deviation of 1.
    - ---> `(features - features.mean(axis=0)) / features.std(axis=0)` <---
- **Missing Features** sometimes we dont have all the features we need.
  - *Solution 1* Fill in missing values with values that are outside features range. Hope is model will learn to ignore that. Putting to 0.
  - *Solution 2* Replace with the mean value over the dataset. This allows us to standardize

## Data Perparation (Training, Validation and Test Data)
- Split our data to training, validation and test data
- **Training Data** Used to train the model. Most important to select feature vectors that represent the parent distribution.
- **Test Data** used to evaluate how well the trained model is doing. Never use the test data when training the model.
- **Validation Data** Not always needed but useful for deep learning. It can help to decide when to stop training and if we are using the correct model. Cant use validation data when reporting actual model performance.

## Partitioning the Dataset
- Typical split is 90% training, 5% validation, 5% testing.
- You can go as low as 1% for Validation/Testing. For classical might want to make the test dataset larger.
- Classical data sometimes dont learn as well. So need more test data. try 80% training, 10% validation & testing or 20% testing.
  - Larger test sets might be appropriate for multiclass model that have classes with low probabilities.

## Steps to produce training, validation and test splits
- **Randomize** the order of the full datase, so classes are evenly mixed
- **Calculate number of samples** in training (ntrn) and valdation (nval) by multiplying the number of samples in the full dataset by the desired fraction. Remaining will fall into the test set
- Assign the first ntrn samples to the training set
- Assign the next nval samples to the validation set
- Assin the remining samples to test set
NOTE: See `4-1.py, 4-2.py`

## K-Fold Cross Validation
- Technique to ensure each sample in the dataset is used at some point for training and testing.
- For small dataset in traditional learning models.
- Helpful to decide between models
- Steps
  1. Pratition full, randomized dataset into "k" non-overlapping groups. Your "k" is arbitrary but ranges from 5-10
  2. Holdback x_0 as test data. Ignore validation for now. 
  3. Holback from of training for new model m_0
  4. Do same and call new model m_1
  5. We end up with mulitple instances of the same type of model trained with different subsets of the full dataset.
  6. Repeate this process for each group (test, validation, training)
  7. We'll end up with "k" models trained with (k-1)/k of the data each, holding 1/k of the data back for testing.
  8. Larger k means more training and less test data. If training time is low, tend toward a larger k.
  9. Evaluate the models individually to get an ide of the how a model trained on a full dataset.
  10. Repeate for each type of model and compare results.
  11. We can start over gain and train teh selected modle type using all of the dataset for training.

## Searching for problems in data
- When summarizing values statistically we look at the **mean** (largest value) and **standard deviation** (smallest value)
- Sort the values from smallest to largest
- **standard error (ofthe mean)** The standard deviation divided by the square root of the number of the values of the dataset
  - It is s the measure of the difference between our mean value and the mean value of the parent distribution
  - we will have a better sense of the parent distribution
  - **we could resonably replace the missing values** by the mean (or median)

- Look at the data. Goal is to represent parent distribution.

## Summary
- Components of dataset : Classes, Labels, Features, Feature Vectors
- Good data set: emphasizing the parent distribution
- Basic data prepartion techniques: 
  - Scale dat
  - dealing with missing features.