# Flashcards Deleted

❌ Some questions do not fit well on flashcards and are therefore removed.

## 1 - Your Deep Learning Journey

- Follow through each cell of the stripped version of the notebook for this chapter. Before executing each cell, guess what will happen.
- Complete the Jupyter Notebook online appendix.

## 2 - From Model to Production

- Create an image recognition model using data you curate, and deploy it on the web.

## 3 - Data Ethics

- In the paper ["Does Machine Learning Automate Moral Hazard and Error"](https://scholar.harvard.edu/files/sendhil/files/aer.p20171084.pdf) why is sinusitis found to be predictive of a stroke?

## 5 - Image Classification

- If you are not familiar with regular expressions, find a regular expression tutorial, and some problem sets, and complete them. Have a look on the book's website for suggestions.
- Look up the documentation for `L` and try using a few of the new methods that it adds.
- Look up the documentation for the Python `pathlib` module and try using a few methods of the `Path` class.
- Calculate the `exp` and `softmax` columns of <<bear_softmax>> yourself (i.e., in a spreadsheet, with a calculator, or in a notebook).

## 8 - Collaborative Filtering Deep Dive

- The "Why" part of "What is a latent factor? Why is it 'latent'?"
- Write the code to create a crosstab representation of the MovieLens data (you might need to do some web searching!).
- Create a class (without peeking, if possible!) and use it.
- The "train a model with it" part of "Rewrite the `DotProduct` class (without peeking, if possible!) and train a model with it."
- What is a good loss function to use for MovieLens? Why?
- What would happen if we used cross-entropy loss with MovieLens? How would we need to change the model?
- What is another name for weight decay?

## 9 - Tabular Modeling Deep Dive

- What is a continuous variable?
- What is a categorical variable?
- Should you pick a random validation set in the bulldozer competition? If no, what kind of validation set should you pick?
- In the section "Creating a Random Forest", just after <<max_features>>, why did `preds.mean(0)` give the same result as our random forest?
- Why do we ensure `saleElapsed` is a continuous variable, even although it has less than 9,000 distinct values?

## 11 - Data Munging with fastai's Mid-Level API

- Write a `Transform` that does the numericalization of tokenized texts (it should set its vocab automatically from the dataset seen and have a `decode` method)
- Why can we easily apply fastai data augmentation transforms to the `SiamesePair` we built?

## 12 - A Language Model from Scratch

- Write a module which predicts the third word given the previous two words of a sentence, without peeking
- Write code to print out the first few batches of the validation set, including converting the token IDs back into English strings, as we showed for batches of IMDb data in <>.
- Experiment with `bernoulli_` to understand how it works.
- Study the refactored version of `LSTMCell` carefully to ensure you understand how and why it does the same thing as the non-refactored version.

## 13 - Data Munging with fastai's Mid-Level API

- "Where does it need to be included in the MNIST CNN? Why?" part of "What is `Flatten`? Where does it need to be included in the MNIST CNN? Why?"
- Run *conv-example.xlsx* yourself and experiment with *trace precedents*
- Have a look at Jeremy or Sylvain's list of recent Twitter "like"s, and see if you find any interesting resources or ideas there.
