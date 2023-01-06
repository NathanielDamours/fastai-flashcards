# 13_convolutions

## What is a *feature*?

A **transformation** of the data which is designed to make it easier for the model to learn from it

## Write out the convolutional kernel matrix for a top edge detector

```py
[-1, -1, -1,
  0,  0,  0,
  1,  1,  1]
```

## What is the value of a convolutional kernel applied to a 3×3 matrix of zeros?

A zero matrix

## What is *padding*?

The additional **pixels** that are **added around** the outside of the image. They allow the kernel to be applied to the edge of the image for a convolution.

## What is *stride*?

It refers to how many pixels at a time the kernel is **moved** during the convolution.

## Create a nested list comprehension

```py
>>> x = [[i*3 + j for j in range(3)] for i in range(3)]
>>> x
[[0, 1, 2], [3, 4, 5], [6, 7, 8]]
```

## What is a *channel*?

It is the **number of activations** per grid cell after a convolution, which is, the size of the second axis of a weight matrix. Channel and feature are often used interchangeably.

## What are the shapes of the `input` and `weight` parameters to PyTorch's 2D convolution?

- `input`: `(minibatch_size, in_channels, i_height, i_width)`
- `weight`: `(out_channels, in_channels, k_height, k_width)`

## How is a convolution related to matrix multiplication?

A convolution operation can be represented as a matrix multiplication.

## What is a "convolutional neural network"?

A Convolutional Neural Network, also known as CNN or ConvNet, is a class of neural networks that specializes in processing data that has a **grid-like topology**, such as an image.

[Source](https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939)

## What are the (4) benefit of refactoring parts of your neural network definition?

- You'll **less** likely get **errors** due to inconsistencies in your architecture.
- It makes it more **obvious** to the reader which **parts of your layers** are actually **changing**.
- It can help improve the **readability** and **maintainability** of your code.
- It can make it easier to **debug** and **troubleshoot** issues in your model.

## What is `nn.Flatten`?

It converts **multi-dimensional tensors into a 1D vector**. It's basically the same as PyTorch's `squeeze` method, but as a module. `nn.Flatten` is useful when you want to transform the multidimensional tensor output by a convolutional layer into a vector, which can be fed into a fully connected (linear) layer.

[Source](https://chat.openai.com/chat)

## What does "NCHW" mean?

It is an **abbreviation** for the axes of the input of the model:

- N: batch size
- C: channels
- H: height
- W: width

## Why does the third layer of the MNIST CNN have `7*7*(1168-16)` multiplications?

There are 1168 parameters for that layer, and ignoring the 16 parameters (=number of filters) of the bias, the (1168-16) parameters is applied to the 7x7 grid.

## What is a "receptive field"?

It's the **area of an image** that is involved in the calculation of a layer.

## What is the size of the receptive field of an activation (size=7, k=7) after two stride-2 convolutions? Why?

$7 x 7$, because the formula for calculating the size of the receptive field after applying multiple convolutions with stride $s$ is:

$size = size + (size - k + 2p) (s - 1)$

where:

- size: size of the receptive field before applying the convolution
- k: kernel size of the convolution
- s: stride of the convolution
- p: padding of the convolution

Therefore,
$size = (7, 7) + ((7, 7) - 7 + 0)(2 -1)$
$= (7, 7) +  ((0, 0) + 0)(2-1)$
$= (7, 7) +  (0, 0) = (7, 7)$

[Source](https://chat.openai.com/chat)

## How is a gray image represented as a tensor?

It is a rank-3 tensor of shape (1, height, width)

## How is a color image represented as a tensor?

It is a rank-3 tensor of shape (3, height, width)

## How does a convolution work with a color input?

The convolutional kernel is of **size `(ch_out, ch_in, ks, ks)`**. For example, with a color input with a kernel size of 3x3 with 7 output channels, that would be (7, 3, 3, 3). The convolution filter for each of the `ch_in=3` channels are applied separately to each of the 3 color channels and summed up, and we have `ch_out` filters like this, giving us a `ch_out` convolutional kernel tensors of size `ch_in=3 x ks x ks`. Thus the final size of this tensor is `(ch_out, ch_in, ks, ks)`. Additionally we would have a bias of size `ch_out`.

## What method can we use to see that data in `DataLoaders`?

`show_batch`

## Why do we double the number of filters after each stride-2 convolution?

Because we're **decreasing** the number of activations in the activation map by a factor of 4; we don't want to decrease the capacity of a layer by too much at a time

## Why do we use a larger kernel in the first conv with MNIST (28x28) (with `simple_cnn`)?

Because this can help the neural network to **learn more effectively**. Indeed, with the first layer, if the kernel size is 3x3, with four output filters, then nine pixels are being used to produce 8 output numbers so there is not much learning since input and output size are almost the same. Neural networks will only create useful features if they're forced to do so—that is, if the number of outputs from an operation is significantly smaller than the number of inputs. To fix this, we can use a larger kernel in the first layer.

## What information does `ActivationStats` save for each layer?

- Mean
- Standard deviation
- Histogram of activations for the specified trainable layers in the model being tracked

## How can we access a learner's callback after training?

They are available with the `Learner` object with the same name as the callback class, but in `snake_case`. For example, the `Recorder` callback is available through `learn.recorder`.

## What are the three statistics plotted by `plot_layer_stats`?

- The mean of the activations
- The standard deviation of the activations
- The percentage of activation near zero

## What does the x-axis represent in `plot_layer_stats`?

The progress of training (batch number)

## Why are activations near zero problematic?

Because it means we have **computation in the model that's doing nothing at all** (since multiplying by zero gives zero). When you have some zeros in one layer, they will therefore generally carry over to the next layer... which will then create more zeros.

## What are the (2) upsides and (2) downsides of training with a larger batch size?

- (+) **More accurate gradients** since they're calculated from more data
- (+) **Faster training** because the model can process more data in each forward and backward pass
- (-) **Less opportunities** for the model to **update weights** because there are fewer batches per epoch
- (-) **Higher memory requirements** because the model has to store activations for more data points at the same time

## Why should we avoid using a high learning rate at the start of training?

Because our **initial weights are not well suited** to the task we're trying to solve. Therefore, it is dangerous to begin training with a high learning rate: we may very well make the training diverge instantly.

## What is 1cycle training?

It's a type of **learning rate schedule** developed by Leslie Smith that combines learning rate warmup and annealing, which allows us to train with higher learning rates.

## What are the (2) benefits of training with a high learning rate?

- **Faster training** — a phenomenon Smith named *super-convergence*.
- **Less overfitting** because we skip over the sharp local minima to end up in a smoother (and therefore more generalizable) part of the loss.

## Why do we want to use a low learning rate at the end of training?

Because it allows us to find the **best part** of loss landscape and further minimize the loss.

## What is "cyclical momentum"?

It suggests that the **momentum varies in the opposite direction of the learning rate**: when we are at high learning rates, we use less momentum, and we use more again in the annealing phase.

## What callback tracks hyperparameter values during training (along with other information)?

The `Recorder` callback

## What does one column of pixels in the `color_dim` plot represent?

The histogram of activations for the specified layer for that batch

## What does "bad training" look like in `color_dim`? Why?

We would see a cycle of dark blue, bright yellow at the bottom return, because this training is not smooth and effectively starts from scratch during these cycles.

## What trainable parameters does a batch normalization layer contain?

- `beta`
- `gamma`

## What allows the trainable parameters of a batch normalization?

They allow the model to have any **mean and variance for each layer**, which are learned during training.

## What statistics are used to normalize in batch normalization during training?

The **mean** and **standard deviation** of the batch.

## What statistics are used to normalize in batch normalization during validation?

The running mean of the statistics calculated during training.

## Why do models with batch normalization layers generalize better?

Because **batch normalization adds some extra randomness** to the training process (most researchers believe that)
