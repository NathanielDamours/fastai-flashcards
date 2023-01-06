# 11_midlevel_data

## What does fastai's *layered* API refer to?

It refers to the **levels of fastai's API**:

- Fastai's high-level API that allows to train neural networks for common applications with just a few lines of code
- Fastai's lower-level APIs that are more flexible and better for custom tasks

## Why does a `Transform` have a `decode` method?

To allow us to **reverse** (if possible) the application of the transform.

## How is `decode` often used ?

To convert predictions and mini-batches into **human**-understandable representation.

## Why does a `Transform` have a `setup` method?

Because, sometimes it is necessary to **initialize some inner state**, like the vocabulary for a tokenizer. The `setup` method handles this.

## How does a `Transform` work when called on a tuple?

The `Transform` is always applied to each item of the tuple. If a type annotation is provided, the `Transform` is only applied to the items with the correct type.

## Which methods do you need to implement when writing your own `Transform`?

Just the `encodes` method, and optionally the `decodes` method for it to be reversible, and `setups` for initializing an inner state.

## What is the operation that allows to normalize the items?

```py
x = (x - x.mean()) - x.std()
```

## Write a `Normalize` transform that fully normalizes items, and that can decode that behavior

```py
class Normalize(Transform):
    def setups(self, items):
        self.mean = items.mean()
        self.std = items.std()

    def encodes(self, items):
        return (items - self.mean) / self.std

    def decodes(self, items):
        return items * self.std + self.mean
```

## What is a `Pipeline`?

The `Pipeline` class is meant for **composing several transforms together**. When you call `Pipeline` on an object, it will automatically call the transforms inside, in order.

```py
>>> tfms = [RandomResizedCrop(224), FlipItem(0.5)]
>>> Pipeline(tfms)
Pipeline: FlipItem -- {'p': 0.5} -> RandomResizedCrop -- {'size': (224, 224), 'min_scale': 0.08, 'ratio': (0.75, 1.3333333333333333), 'resamples': (<Resampling.BILINEAR: 2>, <Resampling.NEAREST: 0>), 'val_xtra': 0.14, 'max_scale': 1.0, 'p': 1.0}
```

## What is a `TfmdLists`?

A **`Pipeline` of `Transform`s** applied to a collection of items.

[Source](https://docs.fast.ai/data.core.html#tfmdlists)

## What is a `Datasets`?

A dataset that creates a tuple from each `tfms` (a list of `Transform`(s) or `Pipeline` to apply).

[Source](https://github.com/fastai/fastai/blob/50c8a760bf4a1bfb2c24ef00b5c100a3d55b4389/fastai/data/core.py#L429)

## How is a `Datasets` different from a `TfmdLists`?

`Datasets` will apply two (or more) pipelines in parallel to the same raw object and build a tuple with the result. This is different from `TfmdLists` which leads to two separate objects for the input and target.

## Why are `TfmdLists` and `Datasets` named with an "s"?

Because they can handle a training and a validation set with the `splits` argument.

## How can you build a `DataLoaders` from a `TfmdLists` or a `Datasets`?

You can call the `dataloaders` method.

## How do you pass `item_tfms` and `batch_tfms` when building a `DataLoaders` from a `TfmdLists` or a `Datasets`?

You can pass `after_item` and `after_batch`, respectively, to the `dataloaders` argument.

## What do you need to do when you want to have your custom items work with methods like `show_batch` or `show_results`?

You need to create a **custom type with a `show` method**, since `TfmdLists`/`Datasets` will decode the items until it reaches a type with a `show` method.
