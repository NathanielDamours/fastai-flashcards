# 09_tabular

## What is the difference between a continuous and a categorical variable?

- **Continuous** variable: have a wide range of "continuous" values (ex: age)
- **Categorical** variable: can take on discrete levels that correspond to different categories (ex: cat and dog)

## Provide (2) of the words that are used for the possible values of a categorical variable

**Levels** or **categories** (ordinal or categorical). For example, movie ratings are ordinal variables and colors are categorical variables.

## What is a "dense layer"?

A layer that is **deeply connected** with its preceding layer which means the neurons of the layer are connected to every neuron of its preceding layer.

[Source](https://analyticsindiamag.com/a-complete-understanding-of-dense-layers-in-neural-networks/)

## How do entity embeddings reduce memory usage and speed up neural networks?

Using entity embeddings allows the data to have a **much more memory-efficient (dense) representation of the data**. This will also lead to speed-ups for the model. On the other hand, especially for large datasets, representing the data as one-hot encoded vectors can be very inefficient (and also sparse).

## What kind of datasets are entity embeddings especially useful for?

Datasets with features that have high levels of **cardinality** (the features have lots of possible categories like ZIP code: 90503). Other methods often overfit to data like this.

## What are the two main families of machine learning algorithms?

- **Ensemble of decision trees**: best for structured data (tabular data)
- **Multilayered neural networks**: best for unstructured data (audio, vision, text, etc.)

## Why do some categorical columns need a special ordering in their classes?

Because ordinal categories may inherently have some order.

## How do you tell a pandas' DataFrame that some categorical columns need a special ordering?

By using `set_categories` with the argument `ordered=True` and passing in the ordered list this information represented in the pandas' DataFrame.

## Summarize what a decision tree algorithm does

It determines **how to group the data based on "questions"** that we ask about the data. That is, we keep splitting the data based on the levels or values of the features and generate predictions based on the average target value of the data points in that group. Here is the algorithm:

1. Loop through each column of the dataset in turn
2. For each column, loop through each possible level of that column in turn
3. Try splitting the data into two groups, based on whether they are greater than or less than that value (or if it is a categorical variable, based on whether they are equal to or not equal to that level of that categorical variable)
4. Find the average sale price for each of those two groups, and see how close that is to the actual sale price of each of the items of equipment in that group. That is, treat this as a very simple "model" where our predictions are simply the average sale price of the item's group
5. After looping through all of the columns and possible levels for each, pick the split point which gave the best predictions using our very simple model
6. We now have two different groups for our data, based on this selected split. Treat each of these as separate datasets, and find the best split for each, by going back to step one for each group
7. Continue this process recursively, and until you have reached some stopping criterion for each group — for instance, stop splitting a group further when it has only 20 items in it.

## Why is a date different from a regular categorical or continuous variable ?

Some dates are **different** to others (ex: some are holidays, weekends, etc.) that cannot be described as just an ordinal variable.

## How can you preprocess a date to allow it to be used in a model?

We can generate many different categorical features about the **properties** of the given date (ex: is it a weekday? is it the end of the month?, etc.)

## What is pickle ?

Pickle is a Python module that is used to **save nearly any Python object as a file**. Indeed, it is used to serialize and deserialize Python objects. Serialization is the process of translating a data structure or object state into a format that can be stored or transmitted and reconstructed later.
The opposite operation, extracting a data structure from a series of bytes, is deserialization

Source:

- [First sentence](https://forums.fast.ai/t/fastbook-chapter-9-questionnaire-solutions-wiki/69932)
- [The rest](https://en.wikipedia.org/wiki/Serialization)

## How are mse, samples, and values calculated in the decision tree drawn in this chapter?

By traversing the tree based on answering questions about the data, we reach the **nodes that tell us** the average value of the data in that group, the mse, and the number of samples in that group.

## How do we deal with outliers in our training data, before building a decision tree?

**You don't have to!** Indeed, in decision tree learning, you do splits based on a metric that depends on the proportions of the classes on the left and right leaves after the split (for instance, Giny Impurity). If there are few outliers (which should be the case: if not, you cannot use any model), then they will not be relevant to these proportions. For this reason, decision trees are robust to outliers.

[Source](https://datascience.stackexchange.com/a/31439)

## How do we handle categorical variables in a decision tree?

We convert the **categorical variables to integers**, where the integers correspond to the discrete levels of the categorical variable. Apart from that, there is nothing special that needs to be done to get it to work with decision trees (unlike neural networks, where we use embedding layers).

## What is bagging?

Train multiple models on random subsets of the data, and use the ensemble of models for prediction.

## What is the difference between `max_samples` and `max_features` when creating a random forest?

- `max_samples` defines how many **samples** we use for each decision tree.
- `max_features` defines how many **features** we use for each decision tree.

Don't forget that when training random forests, we train multiple decision trees on random subsets of the data.

## If you increase `n_estimators` to a very high value, can that lead to overfitting? Why?

No, because the trees added due to the increase of `n_estimators` are independent of each other.

## What is *out of bag error*?

Using only the models not trained on the row of data when going through the data and evaluating the dataset. No validation set is needed.

## Tell (2) reasons why a model's validation set error might be worse than the OOB error

- The model does not **generalize** well.
- The possibility that the validation data has a slightly different **distribution** than the data the model was trained on.

## Why random forests are well suit to show how confident we are in our projections using a particular row of data?

Because you just have to look at the standard deviation between the estimators.

## Why random forests are well suit for predicting with a particular row of data, what were the most important factors, and how did they influence that prediction?

Because you just have to use the `treeinterpreter` package to check how the prediction changes as it goes through the tree, adding up the contributions from each split/feature. Use waterfall plot to visualize.

## Why random forests are well suit to show which columns are the strongest predictors?

Because you just have to look at **feature importance**

## Why random forests are well suit to show how do predictions vary, as we vary these columns?

Look at partial dependence plots

## What's the purpose of removing unimportant variables?

Sometimes, it is better to have a more **interpretable** model with less features, so removing unimportant variables helps in that regard.

## What's a good type of plot for showing tree interpreter results?

Waterfall plot

## What is the *extrapolation problem* ?

It is a problem encountered when it is hard for a model to extrapolate to data that's **outside the domain** of the training data.

## How can you tell if your test or validation set is distributed in a different way to your training set?

We can do so by **training a model to classify if the data is training or validation data**. If the data is of different distributions (out-of-domain data), then the model can properly classify between the two datasets.

## What is boosting?

We train a model that **underfits** the dataset, and train subsequent models that predicts the **error** of the original model. We then add the predictions of all the models to get the **final** prediction.

## How could we use embeddings with a random forest?

Instead of passing in the raw categorical columns, the entity embeddings can be passed **into the random forest** model.

## Does using embeddings improve the performance of a random forest?

Entity embeddings contains **richer representations** of the categorical features and definitely can improve the performance of other models like random forests.

## Why might we not always use a neural net for tabular modeling?

Because they are the **hardest to train and longest to train, and less well-understood**. Instead, random forests should be the first choice/baseline, and neural networks could be tried to improve these results or add to an ensemble.
