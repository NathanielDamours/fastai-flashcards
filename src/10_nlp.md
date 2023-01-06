# 10_nlp

## What is self-supervised learning?

Training a model **without** the use of **labels**. An example is a language model.

## What is a language model?

A language model is a self-supervised model that tries to predict the next **word** of a given passage of text.

## Why is a language model considered self-supervised learning?

Because there are **no labels** (ex: sentiment) provided during training. Instead, the model learns to predict the next word by reading lots of provided text with no labels.

## What are self-supervised models usually used for?

Often, they are used as a **pre-trained model for transfer learning**. However, sometimes, they are used by themselves. For example, a language model can be used for autocomplete algorithms!

## When do we fine-tune language models?

When we want to use a pre-trained language model on a **slightly different corpus** than the one for the current task. Indeed, we need to fine-tune the language model on the corpus of the desired downstream task in order to get a better performance.

## What are the three steps to create a state-of-the-art text classifier?

1. **Train** a language model on a large corpus of text (already done for ULM-FiT by Sebastian Ruder and Jeremy!)
2. **Fine-tune** the language model on text classification dataset
3. **Fine-tune** the language model as a text classifier instead.

## How do the 50,000 unlabeled movie reviews help create a better text classifier for the IMDb dataset?

By allowing the model to learn how to predict the next word of a movie review. As a result, the model better understands the language style and structure of the text classification dataset. Therefore, the model can perform better when fine-tuned as a classifier.

## What are the three steps to prepare your data for a language model?

1. Tokenization
2. Numericalization
3. Language model DataLoader

## What is tokenization?

The process of converting **text into a list of words**.

## Why do we need tokenization?

Because, we need a tokenizer that deals with **complicated cases like punctuation, hyphenated words**... Indeed, converting text into a list of words, it is not as simple as splitting on the spaces.

## Name three different approaches to tokenization

1. Word-based tokenization
2. Subword-based tokenization
3. Character-based tokenization

## What is `xxbos`?

This is a special token added by fastai that indicated the **beginning** of the text.

## List (4/8) rules that fastai applies to text during tokenization

- `fix_html`: replace special HTML characters by a readable version
- `replace_rep`: replace any character repeated three times or more by a special token for repetition (`xxrep`), the number of times it's repeated, then the character
- `replace_wrep`: replace any word repeated three times or more by a special token for word repetition (`xxwrep`), the number of times it's repeated, then the word
- `spec_add_spaces`: add spaces around / and #
- `rm_useless_spaces`: remove all repetitions of the space character
- `replace_all_caps`: lowercase a word written in all caps and adds a special token for all caps (`xxcap`) in front of it
- `replace_maj`: lowercase a capitalized word and adds a special token for capitalized (`xxmaj`) in front of it
- `lowercase`: lowercase all text and adds a special token at the beginning (`xxbos`) and/or the end (`xxeos`)

## Why are repeated characters replaced with a token showing the number of repetitions, and the character that's repeated?

Because it allows the model's embedding matrix **to encode information about general concepts** such as repeated characters which could have special or different meaning than just a single character.

## What is numericalization?

This refers to the mapping of the **tokens to integers** to be passed into the model.

## Why might there be words that are replaced with the *unknown word* token?

Because the embedding matrix would be very **large**, would increase **memory** usage, and would **slow** down training if all the words in the dataset have a token associated with them. Therefore, only words with more than `min_freq` occurrence are assigned a token and finally a number, while others are replaced with the *unknown word* token.

## What does **first row** of the *first batch* contain (in case, with batch size of 64, the first row of the tensor representing the first batch contains the first 64 tokens for the dataset)?

The *beginning* of the **first mini-stream** (tokens 1-64)

Explanation :

1. The dataset is split into 64 mini-streams (batch size)
2. Each batch has 64 rows (batch size) and 64 columns (sequence length)
3. The **first row** of the *first batch* contains the *beginning* of the **first mini-stream** (tokens 1-64)
4. The **second row** of the *first batch* contains the *beginning* of the **second mini-stream**
5. The **first row** of the *second batch* contains the *second chunk* of the **first mini-stream** (tokens 65-128)

## What does the **second row** of the *first batch* contain (in case, with batch size of 64, the first row of the tensor representing the first batch contains the first 64 tokens for the dataset)?

The *beginning* of the **second mini-stream**

Explanation :

1. The dataset is split into 64 mini-streams (batch size)
2. Each batch has 64 rows (batch size) and 64 columns (sequence length)
3. The **first row** of the *first batch* contains the *beginning* of the **first mini-stream** (tokens 1-64)
4. The **second row** of the *first batch* contains the *beginning* of the **second mini-stream**
5. The **first row** of the *second batch* contains the *second chunk* of the **first mini-stream** (tokens 65-128)

## What does the **first row** of the *second batch*  contain (in case, with batch size of 64, the first row of the tensor representing the first batch contains the first 64 tokens for the dataset)?

The *second chunk* of the **first mini-stream** (tokens 65-128)

Explanation :

1. The dataset is split into 64 mini-streams (batch size)
2. Each batch has 64 rows (batch size) and 64 columns (sequence length)
3. The **first row** of the *first batch* contains the *beginning* of the **first mini-stream** (tokens 1-64)
4. The **second row** of the *first batch* contains the *beginning* of the **second mini-stream**
5. The **first row** of the *second batch* contains the *second chunk* of the **first mini-stream** (tokens 65-128)

## Why do we need padding for text classification?

Because we have to **collate the batch** since the documents have **variable sizes**. Other approaches. like cropping or squishing, either to negatively affect training or do not make sense in this context. Therefore, padding is used.

## Why don't we need padding for language modeling?

It is not required for language modeling since the documents are all **concatenated**.

## What does an embedding matrix for NLP contain?

It contains vector representations of **all tokens** in the vocabulary.

## What is shape of an embedding matrix for NLP?

`vocab_size` x `embedding_size`, where `vocab_size` is the length of the vocabulary, and `embedding_size` is an arbitrary number defining the number of latent factors of the tokens.

## What is perplexity?

The **exponential of the loss**. Also, it's a commonly used metric in NLP for language models.

## Why do we have to pass the vocabulary of the language model to the classifier data block?

Because it ensures the same **correspondence of tokens to index** so the model can appropriately use the embeddings learned during LM fine-tuning.

## What is gradual unfreezing?

This refers to unfreezing one layer at a time and fine-tuning the pre-trained model.

## Why is text generation always likely to be ahead of automatic identification of machine generated texts?

Because the classification models could be used to **improve** text generation algorithms (evading the classifier) so the text generation algorithms will always be ahead.
