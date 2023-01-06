# 06_multicat

## How could multi-label classification improve the usability of the bear classifier?

This would **allow for the classification of no bears present**. Otherwise, a multi-class classification model will predict the presence of a bear even if it's not there (unless a separate class is explicitly added).

## How do we encode the dependent variable in a multi-label classification problem?

This is encoded **as a one-hot encoded vector**. Essentially, this means we have a zero vector of the same length of the number of classes, and ones are present at the indices for the classes that are present in the data.

## How do you access the rows and columns of a DataFrame as if it was a matrix?

You can use `.iloc`. For example, `my_dataframe.iloc[10, 10]` will select the element in the 10th row and 10th column as if the DataFrame is a matrix. `iloc` stands for index location.

## How do you get a column by name from a DataFrame

`my_dataframe["column_name"]`. However, a good practice would be to use `my_dataframe.loc["column_name"]` for [explicitness](https://stackoverflow.com/a/38886211) and for [speed](https://stackoverflow.com/a/65875826).

## What is the difference between a `Dataset` and `DataLoader`?

- `Dataset` is a collection which returns a tuple of your independent and dependent variable for a single item.
- `DataLoader` is an extension of the `Dataset` functionality. It is an iterator which provides a stream of mini-batches, where each mini-batch is a couple of a batch of independent variables and a batch of dependent variables.

## What does a Datasets object normally contain?

- Training set
- Validation set

## What does a DataLoaders object normally contain?

- Training dataloader
- Validation dataloader

## What does lambda do in Python?

Lambda is a **shortcut for writing functions** (writing one-liner functions). It is great for quick prototyping and iterating, but since it is not serializable, it cannot be used in deployment and production.

```py
>>> double = lambda x : 2*x
>>> double(4)
8
```

## What are the methods to customise how the independent and dependent variables are created with the data block API?

- `get_x`: specify how the independent variables are created
- `get_y`: specify how the data is labelled

## Why is softmax not an appropriate output activation function when using a one hot encoded target?

Softmax wants to make the model predict **only a single class**, which may not be true in a multi-label classification problem. In multi-label classification problems, the input data could have multiple labels or even no labels.

## Why is `nll_loss` not an appropriate loss function when using a one hot encoded target?

Because `NLLLoss` expects the index representation of the labels. You could convert one-hot targets this way:

```py
>>> target = torch.Tensor([[1, 0, 0, 1], [0, 0, 0, 1], [0, 0, 1, 0]])
>>> target = torch.argmax(target, axis=1)
>>> target
tensor([0, 3, 2])
```

[Source](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html)

## What is the difference between nn.BCELoss and nn.BCEWithLogitsLoss?

- `nn.BCELoss` does not include the initial sigmoid. It assumes that the appropriate activation function (ie. the sigmoid) has already been applied to the predictions.
- `nn.BCEWithLogitsLoss` does both the sigmoid and cross entropy in a single function.

## Why can't we use regular accuracy in a multi-label problem?

Because the regular accuracy function **assumes that the final model-predicted class is the one with the highest activation**. However, in multi-label problems, there can be multiple labels. Therefore, a threshold for the activations needs to be set for choosing the final predicted classes based on the activations, for comparing to the target classes.

## When is it okay to tune an hyperparameter on the validation set?

When the relationship between the hyperparameter and the metric being observed is smooth in order to avoid to pick an inappropriate outlier.

## How is `y_range` implemented in fastai?

`y_range` is implemented using `sigmoid_range` in fastai.

```py
def sigmoid_range(x, low, high):
    return x.sigmoid() * (high-low) + low
```

## What is a regression problem?

A problem in which the labels that are **continuous** values

## What loss function should you use for such a regression problem?

The **mean squared error** loss function

## What do you need to do to make sure the fastai library applies the same data augmentation to your inputs images and your target point coordinates?

You need to use the **correct DataBlock**. In this case, it is the `PointBlock`. This DataBlock automatically handles the application data augmentation to the input images and the target point coordinates.
