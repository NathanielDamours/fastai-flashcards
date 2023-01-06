# Progress

⁉️ The remaining cards to include or to complete.

## 7 - Training a State-of-the-Art Model

- What is the difference between ImageNet and Imagenette? When is it better to experiment on one versus the other?
- What is normalization?
- Why didn't we have to care about normalization when using a pretrained model?
- What is progressive resizing?
- Implement progressive resizing in your own project. Did it help?
- What is test time augmentation? How do you use it in fastai?
- Is using TTA at inference slower or faster than regular inference? Why?
- What is Mixup? How do you use it in fastai?
- Why does Mixup prevent the model from being too confident?
- Why does training with Mixup for five epochs end up worse than training without Mixup?
- What is the idea behind label smoothing?
- What problems in your data can label smoothing help with?
- When using label smoothing with five categories, what is the target associated with the index 1?
- What is the first step to take when you want to prototype quick experiments on a new dataset?

## 9 - Tabular Modeling Deep Dive

- Make a list of reasons why a model's validation set error might be worse than the OOB error. How could you test your hypotheses? # TODO: add more reasons and add the test part. The question is renamed "Tell (2) reasons why a model's validation set error might be worse than the OOB error"

## 12 - A Language Model from Scratch

- Why should we get better results in an RNN if we call `detach` less often? Why might this not happen in practice with a simple RNN?

## 14 - ResNets

- How did we get to a single vector of activations in the CNNs used for MNIST in previous chapters? Why isn't that suitable for Imagenette?
- What do we do for Imagenette instead?
- What is "adaptive pooling"?
- What is "average pooling"?
- Why do we need `Flatten` after an adaptive average pooling layer?
- What is a "skip connection"?
- Why do skip connections allow us to train deeper models?
- What does <<resnet_depth>> show? How did that lead to the idea of skip connections?
- What is "identity mapping"?
- What is the basic equation for a ResNet block (ignoring batchnorm and ReLU layers)?
- What do ResNets have to do with residuals?
- How do we deal with the skip connection when there is a stride-2 convolution? How about when the number of filters changes?
- How can we express a 1×1 convolution in terms of a vector dot product?
- Create a `1x1 convolution` with `F.conv2d` or `nn.Conv2d` and apply it to an image. What happens to the `shape` of the image?
- What does the `noop` function return?
- Explain what is shown in <<resnet_surface>>.
- When is top-5 accuracy a better metric than top-1 accuracy?
- What is the "stem" of a CNN?
- Why do we use plain convolutions in the CNN stem, instead of ResNet blocks?
- How does a bottleneck block differ from a plain ResNet block?
- Why is a bottleneck block faster?
- How do fully convolutional nets (and nets with adaptive pooling in general) allow for progressive resizing?

## 15 - Application Architectures Deep Dive

- What is the "head" of a neural net?
- What is the "body" of a neural net?
- What is "cutting" a neural net? Why do we need to do this for transfer learning?
- What is `model_meta`? Try printing it to see what's inside.
- Read the source code for `create_head` and make sure you understand what each line does.
- Look at the output of `create_head` and make sure you understand why each layer is there, and how the `create_head` source created it.
- Figure out how to change the dropout, layer size, and number of layers created by `cnn_learner`, and see if you can find values that result in better accuracy from the pet recognizer.
- What does `AdaptiveConcatPool2d` do?
- What is "nearest neighbor interpolation"? How can it be used to upsample convolutional activations?
- What is a "transposed convolution"? What is another name for it?
- Create a conv layer with `transpose=True` and apply it to an image. Check the output shape.
- Draw the U-Net architecture.
- What is "BPTT for Text Classification" (BPT3C)?
- How do we handle different length sequences in BPT3C?
- Try to run each line of `TabularModel.forward` separately, one line per cell, in a notebook, and look at the input and output shapes at each step.
- How is `self.layers` defined in `TabularModel`?
- What are the five steps for preventing over-fitting?
- Why don't we reduce architecture complexity before trying other approaches to preventing overfitting?

## 16 - The Training Process

- What is the equation for a step of SGD, in math or code (as you prefer)?
- What do we pass to `cnn_learner` to use a non-default optimizer?
- What are optimizer callbacks?
- What does `zero_grad` do in an optimizer?
- What does `step` do in an optimizer? How is it implemented in the general optimizer?
- Rewrite `sgd_cb` to use the `+=` operator, instead of `add_`.
- What is "momentum"? Write out the equation.
- What's a physical analogy for momentum? How does it apply in our model training settings?
- What does a bigger value for momentum do to the gradients?
- What are the default values of momentum for 1cycle training?
- What is RMSProp? Write out the equation.
- What do the squared values of the gradients indicate?
- How does Adam differ from momentum and RMSProp?
- Write out the equation for Adam.
- Calculate the values of `unbias_avg` and `w.avg` for a few batches of dummy values.
- What's the impact of having a high `eps` in Adam?
- Read through the optimizer notebook in fastai's repo, and execute it.
- In what situations do dynamic learning rate methods like Adam change the behavior of weight decay?
- What are the four steps of a training loop?
- Why is using callbacks better than writing a new training loop for each tweak you want to add?
- What aspects of the design of fastai's callback system make it as flexible as copying and pasting bits of code?
- How can you get the list of events available to you when writing a callback?
- Write the `ModelResetter` callback (without peeking).
- How can you access the necessary attributes of the training loop inside a callback? When can you use or not use the shortcuts that go with them?
- How can a callback influence the control flow of the training loop.
- Write the `TerminateOnNaN` callback (without peeking, if possible).
- How do you make sure your callback runs after or before another callback?

## 17 - A Neural Net from the Foundations

- Write the Python code to implement a single neuron.
- Write the Python code to implement ReLU.
- Write the Python code for a dense layer in terms of matrix multiplication.
- Write the Python code for a dense layer in plain Python (that is, with list comprehensions and functionality built into Python).
- What is the "hidden size" of a layer?
- What does the `t` method do in PyTorch?
- Why is matrix multiplication written in plain Python very slow?
- In `matmul`, why is `ac==br`?
- In Jupyter Notebook, how do you measure the time taken for a single cell to execute?
- What is "elementwise arithmetic"?
- Write the PyTorch code to test whether every element of `a` is greater than the corresponding element of `b`.
- What is a rank-0 tensor? How do you convert it to a plain Python data type?
- What does this return, and why? `tensor([1,2]) + tensor([1])`
- What does this return, and why? `tensor([1,2]) + tensor([1,2,3])`
- How does elementwise arithmetic help us speed up `matmul`?
- What are the broadcasting rules?
- What is `expand_as`? Show an example of how it can be used to match the results of broadcasting.
- How does `unsqueeze` help us to solve certain broadcasting problems?
- How can we use indexing to do the same operation as `unsqueeze`?
- How do we show the actual contents of the memory used for a tensor?
- When adding a vector of size 3 to a matrix of size 3×3, are the elements of the vector added to each row or each column of the matrix? (Be sure to check your answer by running this code in a notebook.)
- Do broadcasting and `expand_as` result in increased memory use? Why or why not?
- Implement `matmul` using Einstein summation.
- What does a repeated index letter represent on the left-hand side of einsum?
- What are the three rules of Einstein summation notation? Why?
- What are the forward pass and backward pass of a neural network?
- Why do we need to store some of the activations calculated for intermediate layers in the forward pass?
- What is the downside of having activations with a standard deviation too far away from 1?
- How can weight initialization help avoid this problem?
- What is the formula to initialize weights such that we get a standard deviation of 1 for a plain linear layer, and for a linear layer followed by ReLU?
- Why do we sometimes have to use the `squeeze` method in loss functions?
- What does the argument to the `squeeze` method do? Why might it be important to include this argument, even though PyTorch does not require it?
- What is the "chain rule"? Show the equation in either of the two forms presented in this chapter.
- Show how to calculate the gradients of `mse(lin(l2, w2, b2), y)` using the chain rule.
- What is the gradient of ReLU? Show it in math or code. (You shouldn't need to commit this to memory—try to figure it using your knowledge of the shape of the function.)
- In what order do we need to call the `*_grad` functions in the backward pass? Why?
- What is `__call__`?
- What methods must we implement when writing a `torch.autograd.Function`?
- Write `nn.Linear` from scratch, and test it works.
- What is the difference between `nn.Module` and fastai's `Module`?

## 18 - CNN Interpretation with CAM

- What is a "hook" in PyTorch?
- Which layer does CAM use the outputs of?
- Why does CAM require a hook?
- Look at the source code of the `ActivationStats` class and see how it uses hooks.
- Write a hook that stores the activations of a given layer in a model (without peeking, if possible).
- Why do we call `eval` before getting the activations? Why do we use `no_grad`?
- Use `torch.einsum` to compute the "dog" or "cat" score of each of the locations in the last activation of the body of the model.
- How do you check which order the categories are in (i.e., the correspondence of index->category)?
- Why are we using `decode` when displaying the input image?
- What is a "context manager"? What special methods need to be defined to create one?
- Why can't we use plain CAM for the inner layers of a network?
- Why do we need to register a hook on the backward pass in order to do Grad-CAM?
- Why can't we call `output.backward()` when `output` is a rank-2 tensor of output activations per image per class?

## 19 - A fastai Learner from Scratch

- What is `glob`?
- How do you open an image with the Python imaging library?
- What does `L.map` do?
- What does `Self` do?
- What is `L.val2idx`?
- What methods do you need to implement to create your own `Dataset`?
- Why do we call `convert` when we open an image from Imagenette?
- What does `~` do? How is it useful for splitting training and validation sets?
- Does `~` work with the `L` or `Tensor` classes? What about NumPy arrays, Python lists, or pandas DataFrames?
- What is `ProcessPoolExecutor`?
- How does `L.range(self.ds)` work?
- What is `__iter__`?
- What is `first`?
- What is `permute`? Why is it needed?
- What is a recursive function? How does it help us define the `parameters` method?
- Write a recursive function that returns the first 20 items of the Fibonacci sequence.
- What is `super`?
- Why do subclasses of `Module` need to override `forward` instead of defining `__call__`?
- In `ConvLayer`, why does `init` depend on `act`?
- Why does `Sequential` need to call `register_modules`?
- Write a hook that prints the shape of every layer's activations.
- What is "LogSumExp"?
- Why is `log_softmax` useful?
- What is `GetAttr`? How is it helpful for callbacks?
- Reimplement one of the callbacks in this chapter without inheriting from `Callback` or `GetAttr`.
- What does `Learner.__call__` do?
- What is `getattr`? (Note the case difference to `GetAttr`!)
- Why is there a `try` block in `fit`?
- Why do we check for `model.training` in `one_batch`?
- What is `store_attr`?
- What is the purpose of `TrackResults.before_epoch`?
- What does `model.cuda` do? How does it work?
- Why do we need to check `model.training` in `LRFinder` and `OneCycle`?
- Use cosine annealing in `OneCycle`.
