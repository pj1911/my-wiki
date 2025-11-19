# Transformers-2 - Natural Language Processing

## Introduction

Transformers can process language data made of words, sentences, and paragraphs. They were first developed for text, but are now state-of-the-art models for many other kinds of input data. Many languages, such as English, are sequences of words separated by white space, together with punctuation symbols making them sequential data. For now we focus only on the words and ignore punctuation.

**Representation.** To use words in a deep neural network, we must first turn each word into numbers. A simple method is to choose a fixed dictionary (vocabulary) of words and represent each word by a vector whose length equals the dictionary size. We then use a one-hot representation: the \(\,k\,\)th word in the dictionary has a vector with a 1 in position \(k\) and 0 in all other positions. For example, if aardwolf is the third word in the dictionary, its vector is \((0, 0, 1, 0, \ldots, 0)\).

**Two issues.** This one-hot scheme has two main problems. First, a realistic dictionary may have several hundred thousand words, so the vectors are very high dimensional. Second, these vectors do not express any similarity or relationship between words. 

## Word embedding

We address both the above issues using word embeddings, which map each word into a dense vector in a lower-dimensional space, typically of a few hundred dimensions. The embedding process uses a matrix \(\mathbf{E}\) of size \(D \times K\), where \(D\)
is the dimensionality of the embedding space and \(K\) is the size of the
dictionary. For each one-hot encoded input vector \(\mathbf{x}_n\) of shape \(K \times 1\), we compute the
embedding vector as:

$$
\mathbf{v}_n = \mathbf{E}\mathbf{x}_n .
$$

Here, \(\mathbf{v}_n \in \mathbb{R}^D\) is the dense embedding (or distributed representation) of the \(n\)-th word in the vocabulary. In particular, words that appear in similar contexts in the corpus tend to have embedding vectors that are close to each other in this \(D\)-dimensional space, so geometric relationships between vectors reflect semantic similarity. Because \(\mathbf{x}_n\) is one-hot, \(\mathbf{v}_n\) of shape \(D \times 1\) is simply the corresponding
column of \(\mathbf{E}\). We learn this \(\mathbf{E}\) from a corpus (a large data set) of text.

### Methods for learning: Word2vec

A simple two-layer neural network. The training set is built by taking a
``window'' of \(M\) adjacent words in the text, with a typical value \(M = 5\).
Each window gives one training sample. The samples are treated as independent,
and the overall error is the sum of the error terms for all samples.

There are two variants. In *continuous bag of words* (CBOW), the target to
be predicted is the middle word, and the remaining *context* words are the
inputs, so the network is trained to 'fill in the blank'. In the
*skip-gram* model, the roles are reversed: the centre word is the input and
the context words are the targets.

This training can be viewed as *self-supervised* learning. The data is a
large corpus of unlabelled text, from which many small windows of word
sequences are sampled at random. The labels come from the text itself by
masking the word whose value the network should predict. After training, the embedding matrix \(\mathbf{E}\) is obtained from the network
weights: it is the transpose of the second-layer weight matrix for the CBOW
model, and the first-layer weight matrix for the skip-gram model. Here is a compact derivation for both CBOW and skip-gram.

**Embedding matrix in CBOW and Skip gram**

**Setup**

Let

- \(K\) = vocabulary size,
- \(D\) = embedding dimension,
- \(\mathbf{x}(w) \in \mathbb{R}^K\) = one-hot input vector for word \(w\),
- \(\mathbf{E} \in \mathbb{R}^{D \times K}\) = embedding matrix,
- \(\mathbf{v}(w) = \mathbf{E}\,\mathbf{x}(w)\) = embedding of word \(w\) (a column of \(\mathbf{E}\)).

#### CBOW

We predict a target word \(w_t\) from its \(M\) context words.

**Architecture**

(i) **First layer (input \(\to\) hidden).**

   Weight matrix \(\mathbf{W}^{(1)} \in \mathbb{R}^{D \times K}\). For one context word \(w\),

$$
\mathbf{h}(w) = \mathbf{W}^{(1)} \mathbf{x}(w).
$$

   For a window of \(M\) context words \(w_1,\dots,w_M\), CBOW uses the average

$$
\mathbf{h} = \frac{1}{M} \sum_{i=1}^M \mathbf{W}^{(1)} \mathbf{x}(w_i).
$$

(ii) **Second layer (hidden \(\to\) output).**

   Weight matrix \(\mathbf{W}^{(2)} \in \mathbb{R}^{K \times D}\). Let \(\mathbf{w}^{(2)}_k{}^{\!\top}\) be the \(k\)th row of \(\mathbf{W}^{(2)}\).
   The logit for predicting word \(k\) is

$$
z_k = \mathbf{w}^{(2)}_k{}^{\!\top} \mathbf{h},
$$

   and the output distribution is

$$
p(k \mid \text{context}) =
  \frac{\exp(z_k)}{\sum_{j=1}^K \exp(z_j)}.
$$

Training adjusts \(\mathbf{W}^{(1)}\) and \(\mathbf{W}^{(2)}\) so that the true
target word \(t\) has high probability.

**Where is the embedding matrix?**

In CBOW we have a hidden vector \(\mathbf{h}\) that summarizes the *context*
words. From this context we must decide which vocabulary word best fills the
blank (the *target* word). For each possible target word \(k\) we therefore want a score that says
“how well does word \(k\) fit this context?  or a similarity measure”. A simple way to get such a similarity measure is to compute a dot product between a vector for the context, \(\mathbf{h}\), and a vector attached to word \(k\), call it \(\mathbf{v}(k)\).

So conceptually we want

$$
z_k = \mathbf{v}(k)^\top \mathbf{h}.
$$

Now look at what the network actually does. The last linear layer has weights
\(\mathbf{W}^{(2)} \in \mathbb{R}^{K \times D}\) and computes

$$
z_k = \mathbf{w}^{(2)}_k{}^{\!\top} \mathbf{h},
$$

where \(\mathbf{w}^{(2)}_k{}^{\!\top}\) is the \(k\)th row of \(\mathbf{W}^{(2)}\).

Compare the two expressions for \(z_k\). They will be identical for all \(\mathbf{h}\) if we simply *define*

$$
\mathbf{v}(k) = \mathbf{w}^{(2)}_k .
$$

Thus the word vector for word \(k\) is exactly the \(k\)th row of
\(\mathbf{W}^{(2)}\). If we stack these word vectors as columns we obtain the
embedding matrix

$$
\mathbf{E} =
\begin{bmatrix}
\mathbf{v}(1) & \cdots & \mathbf{v}(K)
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{w}^{(2)}_1 & \cdots & \mathbf{w}^{(2)}_K
\end{bmatrix}
= \mathbf{W}^{(2)\top}.
$$

So in CBOW the “learned embeddings” are
the parameters of the output layer that connect the context representation
\(\mathbf{h}\) to each possible target word.

#### Skip-gram

Lets assume we have a corpus \(w_1, w_2, \dots, w_T\). At each position \(t\) we treat \(w_t\) as
the *centre* word. With window size \(M\) (e.g.\(M=5\)), the *context*
of \(w_t\) is

$$
w_{t+j} \quad \text{for} \quad j \in \{-M,\dots,-1,1,\dots,M\},
$$

restricted to indices in \(\{1,\dots,T\}\). For each valid \(j\) we form a training
pair

$$
(\text{input } w_t,\; \text{target } w_{t+j}),
$$

using all such pairs or a random subset we perform the following operations.

**Architecture**

(i) **First layer (input \(\to\) hidden).**

   Weight matrix \(\mathbf{W}^{(1)} \in \mathbb{R}^{D \times K}\).

   Input is the one-hot vector \(\mathbf{x}(w_t)\) with shape \(K \times 1\) for the centre word \(w_t\).
   The hidden vector for the input (center) word is given by:

$$
\mathbf{h} = \mathbf{W}^{(1)} \mathbf{x}(w_t).
$$

   Because \(\mathbf{x}(w_t)\) has a single 1 and all zeros, \(\mathbf{h}\) is the \(t\)-th column
   of \(\mathbf{W}^{(1)}\) and has the shape \(D \times 1\). This give a \(D\) dimensional representation vector for the input word.

(ii) **Second layer (hidden \(\to\) output).**

   Weight matrix \(\mathbf{W}^{(2)} \in \mathbb{R}^{K \times D}\).

   For all context position \(j\) we use the same hidden vector
   \(\mathbf{h} \in \mathbb{R}^D\) and the same \(\mathbf{W}^{(2)}\). Let
   \(\mathbf{w}^{(2)}_k{}^{\!\top}\) be the \(k\)th row of \(\mathbf{W}^{(2)}\),
   so \(\mathbf{w}^{(2)}_k \in \mathbb{R}^D\) is a feature vector for
   candidate word \(k\). The logit (a scalar) is

$$
z_k = \mathbf{w}^{(2)}_k{}^{\!\top} \mathbf{h},
$$

   We compute their dot product, so if these two vectors point in a similar direction (high similarity in context),
   \(z_k\) is large and word \(k\) becomes more likely, given the center or target word.

   Collecting all \(z_k\) into \(\mathbf{z} \in \mathbb{R}^K\) gives a single
   distribution

$$
p(k \mid w_t) = \text{softmax}_k(\mathbf{z}),
$$

   which depends only on the centre word \(w_t\). Each context position \(j\)
   uses this same distribution but has its own target label \(w_{t+j}\).

Training adjusts \(\mathbf{W}^{(1)}\) and \(\mathbf{W}^{(2)}\) so that, for every
pair \((w_t, w_{t+j})\) with
\(j \in \{-M,\dots,-1,1,\dots,M\}\), the probability \(p(k \mid w_t)\) is high
when \(k\) is the index of the true context word \(w_{t+j}\). Thus the model learns
a distribution \(p(\cdot \mid w_t)\) that places high mass on all typical
neighbours of \(w_t\).

**What skip-gram actually models**

Skip-gram does *not* model

$$
p(w_{t+j} \mid w_t, j).
$$

It models

$$
p(c \mid w_t),
$$

where \(c\) is any word appearing within the window around \(w_t\). For each centre
position \(t\) and each valid offset \(j \neq 0\) we create an independent pair

$$
(\text{input} = w_t,\; \text{target} = w_{t+j}).
$$

**Why this still learns useful embeddings**

For a fixed centre word \(w_t\), the model sees many targets \(w_{t+j}\) sampled
from words that tend to occur near \(w_t\). Gradients move the centre embedding \(\mathbf{v}(w_t)\) closer (dot product)
to the output vectors of these neighbours. If two words \(w\) and \(w'\) share similar sets of neighbours, they receive
similar updates and end up with similar embeddings.

Note that, the model does not answer “what word is at offset \(j=-2\)?”. It answers “which
words are likely to appear near this centre word?”. Skip-gram is built to learn
from co-occurrence patterns, not exact positions.

**Why the same distribution for all \(j\) is reasonable**

Using the same \(\mathbf{h}\) and \(\mathbf{W}^{(2)}\) for all \(j\) means

$$
p(\cdot \mid w_t) \text{ is the same for every context position.}
$$

This matches the modelling choice: the \(M\) context positions are treated as an
unordered bag of neighbours. Distance information is discarded; the goal is to
capture “which words tend to co-occur”, not “where they appear”.

**Where is the embedding matrix now?**

In skip-gram the network only represents the input word \(w_t\) by the hidden
vector \(\mathbf{h}\). Thus we define the embedding of \(w_t\) as

$$
\mathbf{v}(w_t) = \mathbf{h}.
$$

The first layer is linear, with weights \(\mathbf{W}^{(1)} \in \mathbb{R}^{D
\times K}\) and one-hot input \(\mathbf{x}(w_t)\):

$$
\mathbf{h} = \mathbf{W}^{(1)} \mathbf{x}(w_t).
$$

Since \(\mathbf{x}(w_t)\) has a single 1 at position \(t\),

$$
\mathbf{h} = \mathbf{W}^{(1)}_{:,t},
$$

so

$$
\mathbf{v}(w_t) = \mathbf{W}^{(1)}_{:,t},
$$

and all word embeddings are the columns of \(\mathbf{W}^{(1)}\). Therefore

$$
\mathbf{E} = \mathbf{W}^{(1)}.
$$

**Summary**

- In CBOW, the embedding of a *predicted* word \(k\) is the vector used
  to score that word at the output. Those vectors are the rows of
  \(\mathbf{W}^{(2)}\), so
  \(\mathbf{E} = \mathbf{W}^{(2)\top}\).

- In skip-gram, the embedding of an *input* word \(k\) is the hidden
  vector produced by its one-hot input. Those vectors are the columns of
  \(\mathbf{W}^{(1)}\), so
  \(\mathbf{E} = \mathbf{W}^{(1)}\).

Words that
are semantically related are mapped to nearby positions in the embedding space.
This happens because related words tend to occur with similar context words
more often than unrelated words. For example, the words 'city' and
'capital' appear more often as context for target words such as 'Paris' or
'London' than for 'orange' or 'polynomial'. The network can then more
easily predict the missing words if 'Paris' and 'London' are mapped to
nearby embedding vectors.

The learned embedding space often has richer semantic structure than simple
closeness of related words, and it supports simple vector arithmetic. For
example, the relation 'Paris is to France as Rome is to Italy' can be
expressed in terms of embedding vectors. Writing \(\mathbf{v}(\text{word})\) for
the embedding of *word*, we find

$$
\mathbf{v}(\text{Paris}) - \mathbf{v}(\text{France})
+ \mathbf{v}(\text{Italy}) \simeq \mathbf{v}(\text{Rome})
$$

Word embeddings were first developed as stand-alone tools for natural language
processing. Today they are more often used as pre-processing steps for deep
neural networks, and can be viewed as the first layer of such a network. The
embedding matrix may be fixed, using some standard pre-trained embeddings, or
treated as an adaptive layer that is learned during end-to-end training of the
whole system. In the adaptive case, the embedding layer can be initialized with
random weights or with a standard pre-trained embedding matrix.

## Tokenization

A fixed dictionary of whole words has problems. It cannot handle unseen or
misspelled words, and it ignores punctuation and other character sequences such
as computer code.

A pure character-level approach fixes these issues by defining the dictionary as
all characters: upper- and lower-case letters, digits, punctuation, and white-space
symbols such as spaces and tabs. But this approach throws away explicit word
structure. The neural network must then learn words from raw characters, and the
sequence becomes much longer, increasing computation.

We can combine the advantages of word- and character-level views by adding a
pre-processing step called *tokenization*. This converts the original string of
words and punctuation symbols into a string of *tokens*. Tokens are short
character sequences. They may be complete common words, fragments of longer
words, or even individual characters, which can be combined to represent rare
words. Tokenization naturally handles punctuation,
computer code, and other symbol sequences. It can also extend to other modalities
such as images. Variants of the same word can share tokens: for example,
'cook', 'cooks', 'cooked', 'cooking', and 'cooker' can all include the token
'cook', so their representations are related.

There are many tokenization methods. One important example is byte pair
encoding (BPE), originally used for data compression and adapted to text by
merging characters instead of bytes. The
procedure is:

1. Start with a token list that contains all individual characters.
2. In a large text corpus, find the most frequent adjacent pair of tokens.
3. Replace each occurrence of this pair with a new, single token.
4. To avoid merging across word boundaries, do not create a new token from
   a pair if the second token begins with a white-space character.
5. Repeat the merge steps

Initially, the number of tokens in the vocabulary equals the number of distinct
characters, which is small. As merges are applied, the vocabulary size grows.
If we continue long enough, the tokens approach whole words. In practice, we
fix a maximum vocabulary size in advance as a compromise between character-
level and word-level representations and stop the algorithm when this size is
reached.

In most deep learning applications to natural language, input text is first mapped
to a tokenized sequence. However, for the rest of this chapter we will use
word-level representations because they make the main ideas easier to present.

## Bag of words

The task is to calculate the likelihood (probability) of a specific sequence of words occurring. We now model the joint distribution \(p(\mathbf{x}_1,\ldots,\mathbf{x}_N)\) of this
ordered sequence of vectors (words/ token in our case). The simplest
assumption is that all words are drawn independently from the same
distribution (ignoring the order). Then

$$
p(\mathbf{x}_1,\ldots,\mathbf{x}_N)
= \prod_{n=1}^N p(\mathbf{x}_n)
$$

The distribution \(p(\mathbf{x})\) is the same for all positions or tokens and can be
represented as a table of probabilities over the dictionary of words or tokens where each word or token has a fixed probability independent of its position.
The maximum-likelihood estimate simply sets each table entry equal to the
fraction of times that word occurs in the training set. This is called a
bag of words model because it ignores word order completely.

We can use the bag-of-words idea to build a simple text classifier. For
instance, in sentiment analysis a review is classified as positive or negative.
The naive Bayes classifier assumes that, within each class \(C_k\), the
words are independent, but that each class has its own distribution. Thus

$$
p(\mathbf{x}_1,\ldots,\mathbf{x}_N \mid C_k)
= \prod_{n=1}^N p(\mathbf{x}_n \mid C_k)
$$

Now apply Bayes’ rule to get the posterior over the class:

$$
p(C_k \mid \mathbf{x}_1,\ldots,\mathbf{x}_N)
= \frac{p(\mathbf{x}_1,\ldots,\mathbf{x}_N \mid C_k)\,p(C_k)}
       {p(\mathbf{x}_1,\ldots,\mathbf{x}_N)}.
$$

Substitute the likelihood:

$$
p(C_k \mid \mathbf{x}_1,\ldots,\mathbf{x}_N)
= \frac{\Bigl[\prod_{n=1}^N p(\mathbf{x}_n \mid C_k)\Bigr]\,p(C_k)}
       {p(\mathbf{x}_1,\ldots,\mathbf{x}_N)}.
$$

The denominator

$$
p(\mathbf{x}_1,\ldots,\mathbf{x}_N)
$$

does not depend on \(k\) (it is the same for all classes), so when we compare or
normalize over \(k\) we can treat it as a constant. Therefore we write

$$
p(C_k \mid \mathbf{x}_1,\ldots,\mathbf{x}_N)
\propto p(C_k)\prod_{n=1}^N p(\mathbf{x}_n \mid C_k),
$$

Both the class-conditional distributions \(p(\mathbf{x}\mid C_k)\) and the priors
\(p(C_k)\) can be estimated from training frequencies. For a new sequence, we
multiply the corresponding table entries to obtain posterior scores. If a word
appears in the test set but never appeared in the training set for a given
class, then its estimated probability for that class is zero, and the whole
product becomes zero. To avoid this, the probability tables are usually
smoothed after training by adding a small amount of probability
uniformly across all entries so that no entry is exactly zero.

## Autoregressive models

A major limitation of the bag-of-words model is that it ignores word order. To
include order we use an *autoregressive* factorization. Without loss of
generality we can write the joint distribution over a sequence as

$$
p(\mathbf{x}_1,\ldots,\mathbf{x}_N)
= \prod_{n=1}^N p(\mathbf{x}_n \mid \mathbf{x}_1,\ldots,\mathbf{x}_{n-1})
$$

Each conditional \(p(\mathbf{x}_n \mid \mathbf{x}_1,\ldots,\mathbf{x}_{n-1})\) could
be stored as a table whose entries are estimated from frequency counts in the
training corpus. But the size of these tables grows exponentially with the
sequence length, so this direct approach is not feasible. Let us try to prove this in the following setup.

**Setup**

Assume a vocabulary of size \(K\) (number of distinct words or tokens), a sequence length \(N\) and each \(x_i\) being a discrete random variable taking one of these \(K\) values:

$$
x_i \in \{1,2,\ldots,K\}.
$$

The full autoregressive factorization is

$$
p(x_1,\ldots,x_N) = \prod_{n=1}^N p(x_n \mid x_1,\ldots,x_{n-1}).
$$

We want to store each conditional \(p(x_n \mid x_1,\ldots,x_{n-1})\) in a table.
Here we are counting how many model parameters (independent
probability values) are needed to specify this conditional distribution. Consider a specific position \(n\) and let us count how big that parameter table
must be.

**1. Counting possible histories \((x_1,\ldots,x_{n-1})\).**

A *history* for position \(n\) is any concrete sequence of values

$$
h = (x_1,\ldots,x_{n-1})
  = (i_1,\ldots,i_{n-1}),
$$

where each \(i_j\) is one of \(\{1,\ldots,K\}\) token.

$$
\{\text{number of possible histories of length } n-1\}
= \underbrace{K \times K \times \cdots \times K}_{n-1 \text{ factors}}
= K^{n-1}.
$$

**2. What is stored for one fixed history?**

Take one specific history

$$
h = (x_1=i_1,\ldots,x_{n-1}=i_{n-1}).
$$

For this history we need the conditional distribution of \(x_n\):

$$
p(x_n \mid h).
$$

The variable \(x_n\) can again be any of the \(K\) vocabulary items. So we must
store \(K\) probabilities, one for each possible value of \(x_n\):

$$
p(x_n = 1 \mid h),\; p(x_n = 2 \mid h),\;\ldots,\; p(x_n = K \mid h).
$$

If we wrote this as a row in a table, that row would contain exactly these \(K\)
numbers. These numbers are exactly the parameters that define the model’s
behaviour for this history.

**3. \(K-1\) free parameters per history.**

For each fixed history \(h\), the \(K\) probabilities must obey:

 *(i) Non-negativity:*

$$
0 \le p(x_n = k \mid h) \le 1
\qquad \text{for all } k=1,\ldots,K.
$$

*(ii) Normalization:*

$$
\sum_{k=1}^K p(x_n = k \mid h) = 1.
$$

The normalization constraint removes one degree of freedom. Once we choose
any \(K-1\) of the probabilities, the last one is forced to make the sum 1:

$$
p(x_n = K \mid h)
= 1 - \sum_{k=1}^{K-1} p(x_n = k \mid h).
$$

So although there are \(K\) numbers in the row, only \(K-1\) of them can be chosen
independently. We say the distribution for this history has \(K-1\) free
parameters.

**4. Total number of parameters for the table at step \(n\).**

- Number of distinct histories \(h\) of length \(n-1\): \(K^{n-1}\).
- Free parameters for each history’s row: \(K-1\)

Therefore, the total number of free parameters needed to specify the whole
conditional table is

$$
\underbrace{K^{n-1}}_{\text{rows (histories)}} \times
\underbrace{(K-1)}_{\text{free params per row}}
= (K-1)K^{n-1}.
$$

This quantity grows proportionally to \(K^{n-1}\), which increases
*exponentially* as \(n\) increases. That is why storing these conditionals as
raw tables quickly becomes infeasible for realistic vocabulary sizes \(K\) and
sequence lengths \(n\).

**5. Total size up to length \(N\).**

To model all positions \(n=1,\ldots,N\), we need all these tables:

$$
\text{total parameters}
= \sum_{n=1}^N (K-1)K^{n-1}
= (K-1)\frac{K^{N}-1}{K-1}
= K^{N}-1.
$$

So the total number of table entries is on the order of \(K^{N}\), which grows
exponentially with the sequence length \(N\). Hence representing the
autoregressive model directly with probability tables is infeasible for realistic
\(K\) and \(N\).

### n-gram

We simplify the model by assuming that the conditional for step \(n\) depends
only on the last \(L\) observations. For example, if \(L=2\) then

$$
p(\mathbf{x}_1,\ldots,\mathbf{x}_N)
= p(\mathbf{x}_1)p(\mathbf{x}_2 \mid \mathbf{x}_1)
  \prod_{n=3}^N p(\mathbf{x}_n \mid \mathbf{x}_{n-1},\mathbf{x}_{n-2})
$$

The conditional distributions \(p(\mathbf{x}_n \mid
\mathbf{x}_{n-1},\mathbf{x}_{n-2})\) are shared across all positions and can be
represented as tables whose entries are estimated from statistics of successive
triplets of words in a corpus. The case with \(L=1\) is a *bi-gram* model, which depends on pairs of adjacent
words. The case \(L=2\) is a *tri-gram* model, involving triplets. More
generally these are called *\(n\)-gram* models.

All the models in this section can be run *generatively* to create text.
For example, given the first two words we can sample the third from
\(p(\mathbf{x}_n \mid \mathbf{x}_{n-1},\mathbf{x}_{n-2})\), then use the second
and third words to sample the fourth, and so on. The resulting text will
usually be incoherent, because each word depends only on a short context. Good
text models must capture long-range dependencies in language. Simply increasing
\(L\) is not practical, because the size of the probability tables grows
exponentially in \(L\), making models beyond tri-grams prohibitively expensive.
Nevertheless, the autoregressive factorization will remain central when we move
to modern language models based on deep neural networks, such as transformers.

One way to extend the effective context for language, without the exponential
number of parameters of \(n\)-gram tables, is to add *latent* (hidden)
variables and use a *hidden Markov model* (HMM).

### HMM for a word sequence (n-gram)

Consider a sequence of \(N\) words (or tokens)

$$
x_1, x_2, \dots, x_N.
$$

For each *position* \(n\) in the sequence we introduce a hidden state \(z_n \in \{ z_1, z_2, \dots, z_N\}\), which can take one of \(S\) discrete values (for example, \(S\) different latent
“topics’’ or parts of speech such as noun, verb, adjective), and an observed
word \(x_n\) that takes one of \(K\) vocabulary values. The model defines:

- **Initial state distribution \(p(z_1)\).**  
  This is a categorical distribution over the \(S\) possible values of
  \(z_1\), so it is a length-\(S\) vector:

$$
p(z_1)
= \big(p(z_1=1),\dots,p(z_1=S)\big), \quad
\sum_{s=1}^S p(z_1=s)=1.
$$

- **Transition distribution \(p(z_n \mid z_{n-1})\).**  
  For each previous state \(i \in \{1,\dots,S\}\) we need a full
  distribution over the next state \(j \in \{1,\dots,S\}\). Thus we have
  \(S\) rows (one per \(i\)) and \(S\) columns (one per \(j\)):

$$
p(z_n \mid z_{n-1})
= \big[p(z_n=j \mid z_{n-1}=i)\big]_{i,j=1}^S,
$$

  an \(S \times S\) matrix whose each row sums to 1.

- **Emission distribution \(p(x_n \mid z_n)\).**  
  For each hidden state \(s\) we need a distribution over all \(K\) words
  \(k \in \{1,\dots,K\}\). So we have \(S\) rows and \(K\) columns:

$$
p(x_n \mid z_n)
= \big[p(x_n=k \mid z_n=s)\big]_{s=1,\dots,S;\;k=1,\dots,K},
$$

  an \(S \times K\) matrix whose each row sums to 1.

Now, we want the joint distribution over all hidden states and observations:

$$
p(x_{1:N}, z_{1:N}) = p(z_1, x_1, z_2, x_2, \dots, z_N, x_N).
$$

This can be done in the following steps:

**1. Chain rule.**

Apply the chain rule in the time order:

$$
\begin{aligned}
p(x_{1:N}, z_{1:N})
&= p(z_1)\,
   p(x_1 \mid z_1)\,
   p(z_2 \mid z_1, x_1)\,
   p(x_2 \mid z_1, x_1, z_2)\,\cdots \\
&\quad \cdots\,
   p(z_N \mid z_{1:N-1}, x_{1:N-1})\,
   p(x_N \mid z_{1:N}, x_{1:N-1}).
\end{aligned}
$$

**2. HMM assumptions.**

An HMM imposes two conditional independence assumptions:

(i) Markov property for hidden states

$$
p(z_n \mid z_{1:n-1}, x_{1:n-1}) = p(z_n \mid z_{n-1})
\quad\text{for } n \ge 2.
$$

(ii) Emission depends only on current state

$$
p(x_n \mid z_{1:n}, x_{1:n-1}) = p(x_n \mid z_n)
\quad\text{for } n \ge 1.
$$

**3. Simplify each factor.**

Apply these to the chain rule factors:

- For \(n=1\):

$$
p(z_1) \quad\text{(unchanged)}, \qquad
p(x_1 \mid z_1) \quad\text{(already of the form } p(x_1 \mid z_1)\text{)}.
$$

- For \(n=2\):

$$
p(z_2 \mid z_1, x_1) = p(z_2 \mid z_1),
$$

$$
p(x_2 \mid z_1, x_1, z_2) = p(x_2 \mid z_2).
$$

- In general, for \(n = 2,\dots,N\):

$$
p(z_n \mid z_{1:n-1}, x_{1:n-1}) = p(z_n \mid z_{n-1}),
$$

$$
p(x_n \mid z_{1:n}, x_{1:n-1}) = p(x_n \mid z_n).
$$

**4. Collect all terms.**

Replacing every factor in the chain rule by its simplified HMM form gives

$$
\begin{aligned}
p(x_{1:N}, z_{1:N})
&= p(z_1)\, p(x_1 \mid z_1)\,
   \prod_{n=2}^N p(z_n \mid z_{n-1})\, p(x_n \mid z_n) \\
&= p(z_1)\,\Bigg[\prod_{n=2}^N p(z_n \mid z_{n-1})\Bigg]\,
           \Bigg[\prod_{n=1}^N p(x_n \mid z_n)\Bigg].
\end{aligned}
$$

$$
p(x_{1:N}, z_{1:N})
= p(z_1)\,\prod_{n=2}^N p(z_n \mid z_{n-1})\,
  \prod_{n=1}^N p(x_n \mid z_n).
$$

Operationally, the model generates a sequence as follows:

(i) Sample the first hidden state

$$
z_1 \sim p(z_1).
$$

(ii) Emit the first word from this state

$$
x_1 \sim p(x_1 \mid z_1).
$$

(iii) For each later position \(n=2,\dots,N\):

$$
\begin{aligned}
  z_n &\sim p(z_n \mid z_{n-1}) && \text{(move to a new hidden state)}\\
  x_n &\sim p(x_n \mid z_n)     && \text{(emit the next word).}
\end{aligned}
$$

This step-by-step process corresponds exactly to the factorization

$$
p(x_{1:N}, z_{1:N})
= p(z_1)\,\prod_{n=2}^N p(z_n \mid z_{n-1})\,
  \prod_{n=1}^N p(x_n \mid z_n).
$$

Since \(p(z_1)\) has \(O(S)\) parameters, the \(S \times S\) transition matrix \(p(z_n \mid z_{n-1})\) has \(O(S^2)\)
parameters, and the \(S \times K\) emission matrix \(p(x_n \mid z_n)\) has \(O(SK)\) parameters, the total number of learnable parameters scales as

$$
O(S^2 + SK),
$$

which depends only on the number of states \(S\) and vocabulary size \(K\), but
*not* on the sequence length \(N\). In this way an HMM can model long
sequences without the \(O(K^L)\) parameter blow-up of an \(L\)-gram table.

**Long-range dependencies.**

We want to see how \(x_n\) can depend on *all* earlier words in an HMM.

$$
p(x_n \mid x_{1:n-1})
= \sum_{z_n} p(x_n, z_n \mid x_{1:n-1})
= \sum_{z_n} p(x_n \mid z_n, x_{1:n-1})\,p(z_n \mid x_{1:n-1}).
$$

**1. By definition of conditional probability.**

$$
p(x_n \mid x_{1:n-1})
= \frac{p(x_n, x_{1:n-1})}{p(x_{1:n-1})}.
$$

**2. Insert the hidden variable by marginalization.**

The joint probability of \((x_n, x_{1:n-1})\) can be written by summing over all
possible values of the hidden state \(z_n\) (from law of total probability):

$$
p(x_n, x_{1:n-1})
= \sum_{z_n} p(x_n, z_n, x_{1:n-1}).
$$

Substitute this into the conditional:

$$
p(x_n \mid x_{1:n-1})
= \frac{1}{p(x_{1:n-1})}
  \sum_{z_n} p(x_n, z_n, x_{1:n-1}).
$$

Now divide inside the sum:

$$
p(x_n \mid x_{1:n-1})
= \sum_{z_n} \frac{p(x_n, z_n, x_{1:n-1})}{p(x_{1:n-1})}
= \sum_{z_n} p(x_n, z_n \mid x_{1:n-1}).
$$

This gives:

$$
p(x_n \mid x_{1:n-1})
= \sum_{z_n} p(x_n, z_n \mid x_{1:n-1}).
$$

**3. Apply the product rule to the conditional joint.**

For any random variables \(A,B,C\),

$$
p(A,B \mid C) = p(A \mid B,C)\,p(B \mid C).
$$

This is just the chain rule applied to the conditional distribution given \(C\):

$$
p(A,B,C)
= p(A \mid B,C)\,p(B \mid C)\,p(C),
$$

and dividing both sides by \(p(C)\) gives the conditional form.

Now take

$$
A = x_n,\quad B = z_n,\quad C = x_{1:n-1}.
$$

Then

$$
p(x_n, z_n \mid x_{1:n-1})
= p(x_n \mid z_n, x_{1:n-1})\,p(z_n \mid x_{1:n-1}).
$$

Substitute this into the previous sum:

$$
\begin{aligned}
p(x_n \mid x_{1:n-1})
&= \sum_{z_n} p(x_n, z_n \mid x_{1:n-1}) \\
&= \sum_{z_n} p(x_n \mid z_n, x_{1:n-1})\,p(z_n \mid x_{1:n-1}).
\end{aligned}
$$

By the HMM emission assumption, the observation depends only on the current
state:

$$
p(x_n \mid z_n, x_{1:n-1}) = p(x_n \mid z_n).
$$

So

$$
p(x_n \mid x_{1:n-1})
= \sum_{z_n} p(x_n \mid z_n)\,p(z_n \mid x_{1:n-1}).
$$

Here \(p(z_n \mid x_{1:n-1})\) is the *belief* over the current hidden state
given all previous words. This belief is updated recursively from
\(p(z_{n-1} \mid x_{1:n-2})\) using the transition and emission distributions. In
principle, every past word \(x_1,\dots,x_{n-1}\) can influence \(p(z_n \mid
x_{1:n-1})\), and therefore \(p(x_n \mid x_{1:n-1})\).

**How new observations overwrite older information.**

We want an explicit formula for our *updated belief* about the current
hidden state after seeing the new observation at time \(t\); this belief is the
filtering distribution

$$
p(z_t \mid x_{1:t}).
$$

**1. Start from Bayes’ rule.**  
Treat \(x_t\) as the new observation and \(x_{1:t-1}\) as given context:
We have:

$$
p(A \mid B,C)
= \frac{p(B \mid A,C)\,p(A \mid C)}{p(B \mid C)}.
$$

Where,

$$
A = z_t,\quad B = x_t,\quad C = x_{1:t-1}.
$$

So

$$
p(z_t \mid x_{1:t-1}, x_t)
= \frac{p(x_t \mid z_t, x_{1:t-1})\,p(z_t \mid x_{1:t-1})}
       {p(x_t \mid x_{1:t-1})}.
$$

Finally, note that

$$
p(z_t \mid x_{1:t})
= p(z_t \mid x_{1:t-1}, x_t),
$$

**2. Use the emission assumption.**  
In an HMM, \(x_t\) depends only on \(z_t\):

$$
p(x_t \mid z_t, x_{1:t-1}) = p(x_t \mid z_t).
$$

So

$$
p(z_t \mid x_{1:t})
= \frac{p(x_t \mid z_t)\,p(z_t \mid x_{1:t-1})}
       {p(x_t \mid x_{1:t-1})}.
$$

The denominator does not depend on \(z_t\), so it is a normalizing constant.
Thus

$$
p(z_t \mid x_{1:t})
\propto p(x_t \mid z_t)\,p(z_t \mid x_{1:t-1}).
$$

**3. Expand the predictive term.**

In the previous step we obtained

$$
p(z_t \mid x_{1:t})
\propto p(x_t \mid z_t)\,p(z_t \mid x_{1:t-1}),
$$

so to complete the update we need an explicit expression for the
*predictive* distribution \(p(z_t \mid x_{1:t-1})\) in terms of quantities
at time \(t-1\). This is where the Markov structure of the HMM enters.

**(i) Start from the definition of the predictive term.**

By definition of conditional probability,

$$
p(z_t \mid x_{1:t-1})
= \frac{p(z_t, x_{1:t-1})}{p(x_{1:t-1})}.
$$

**(ii) Introduce \(z_{t-1}\) by marginalization.**

Use the law of total probability on the joint:

$$
p(z_t, x_{1:t-1})
= \sum_{z_{t-1}} p(z_t, z_{t-1}, x_{1:t-1}).
$$

Substitute into the conditional:

$$
p(z_t \mid x_{1:t-1})
= \frac{1}{p(x_{1:t-1})}
  \sum_{z_{t-1}} p(z_t, z_{t-1}, x_{1:t-1}).
$$

Bring the constant denominator inside the sum:

$$
p(z_t \mid x_{1:t-1})
= \sum_{z_{t-1}} \frac{p(z_t, z_{t-1}, x_{1:t-1})}{p(x_{1:t-1})}
= \sum_{z_{t-1}} p(z_t, z_{t-1} \mid x_{1:t-1}).
$$

This gives

$$
p(z_t \mid x_{1:t-1})
= \sum_{z_{t-1}} p(z_t, z_{t-1} \mid x_{1:t-1}).
$$

**(iii) Apply the product rule inside the sum.**

For any variables \(A,B,C\),

$$
p(A,B \mid C) = p(A \mid B,C)\,p(B \mid C).
$$

Let

$$
A = z_t,\quad B = z_{t-1},\quad C = x_{1:t-1}.
$$

Then

$$
p(z_t, z_{t-1} \mid x_{1:t-1})
= p(z_t \mid z_{t-1}, x_{1:t-1})\,
  p(z_{t-1} \mid x_{1:t-1}).
$$

Substitute back:

$$
\begin{aligned}
p(z_t \mid x_{1:t-1})
&= \sum_{z_{t-1}} p(z_t, z_{t-1} \mid x_{1:t-1}) \\
&= \sum_{z_{t-1}}
   p(z_t \mid z_{t-1}, x_{1:t-1})\,
   p(z_{t-1} \mid x_{1:t-1}).
\end{aligned}
$$

This is the desired expression of the predictive term in terms of the previous
belief \(p(z_{t-1} \mid x_{1:t-1})\) and the state dynamics.

**4. Use the Markov assumption.**  
In an HMM, the next state depends only on the previous state:

$$
p(z_t \mid z_{t-1}, x_{1:t-1}) = p(z_t \mid z_{t-1}).
$$

So

$$
p(z_t \mid x_{1:t-1})
= \sum_{z_{t-1}} p(z_t \mid z_{t-1})\,p(z_{t-1} \mid x_{1:t-1}).
$$

**5. Combine the pieces.**

Substitute the expression for \(p(z_t \mid x_{1:t-1})\) back into the proportional
form:

$$
\begin{aligned}
p(z_t \mid x_{1:t})
&\propto p(x_t \mid z_t)\,p(z_t \mid x_{1:t-1}) \\
&= p(x_t \mid z_t)
   \sum_{z_{t-1}} p(z_t \mid z_{t-1})\,p(z_{t-1} \mid x_{1:t-1}).
\end{aligned}
$$

This gives

$$
p(z_t \mid x_{1:t})
\propto p(x_t \mid z_t)
        \sum_{z_{t-1}} p(z_t \mid z_{t-1})\,p(z_{t-1} \mid x_{1:t-1}),
$$

with the proportionality constant chosen so that
\(\sum_{z_t} p(z_t \mid x_{1:t}) = 1\).
This is the standard forward (filtering) update: the belief over \(z_t\) after
seeing \(x_t\).

It is often useful to view this update at time \(t+1\) as two steps:

(i) **Prediction (from \(t\) to \(t+1\)):**

$$
\tilde{p}(z_{t+1} \mid x_{1:t})
= \sum_{z_t} p(z_{t+1} \mid z_t)\,p(z_t \mid x_{1:t}),
$$

   which uses only the transition probabilities and the previous belief.

(ii) **Correction (using \(x_{t+1}\)):**

$$
p(z_{t+1} \mid x_{1:t+1})
\propto p(x_{t+1} \mid z_{t+1})\,
         \tilde{p}(z_{t+1} \mid x_{1:t}),
$$

$$
p(z_{t+1} \mid x_{1:t+1})
\propto p(x_{t+1} \mid z_{t+1})\,
          \sum_{z_t} p(z_{t+1} \mid z_t)\,p(z_t \mid x_{1:t}),
$$

   which reweights the predicted belief using the likelihood of the new
   observation.

Thus, all past observations \(x_{1:t}\) affect \(p(z_{t+1} \mid x_{1:t+1})\) only
through the current summary \(p(z_t \mid x_{1:t})\). At each new time step this
summary is first mixed by the transition probabilities and then reshaped by the
new word. After many such updates, different early histories produce almost the
same state distribution. In practice, this means that the influence of very old
observations on future predictions becomes very small very quickly.

## Recurrent neural networks

\(n\)-gram models scale badly with sequence length because they store large,
unstructured tables of conditional probabilities. We can get much better scaling
by using parameterized models based on neural networks. But if we try to apply a standard feed-forward network directly to word sequences,
two problems appear:

- The network expects a fixed number of inputs and outputs, but real
  sequences have variable length in both training and test data.
- A word (or phrase) that appears in different positions should usually
  represent the same concept, but a plain feed-forward network would
  treat each position with separate parameters.

Ideally, we want an architecture that:

1. shares parameters across all positions in the sequence (an equivariance property), and
2. can handle sequences of different lengths.

To achieve this, we take inspiration from the hidden Markov model and introduce
a hidden state \(z_n\) for each step \(n\) in the sequence. At each step the network
takes as input

$$
(x_n, z_{n-1})
$$

where \(x_n\) is the current word and \(z_{n-1}\) is the previous hidden state, and
it outputs

$$
(y_n, z_n),
$$

where \(y_n\) is the network output at that step (for example, a predicted word)
and \(z_n\) is the updated hidden state. Instead of building a different network for each time step, we use *one*
neural network cell and apply it repeatedly along the sequence. Formally, if the cell has parameters
\(\theta\) (weights and biases), then at every step \(n\) we compute

$$
(z_n, y_n) = f_\theta(x_n, z_{n-1}),
$$

using the *same* \(\theta\) for all \(n\). The
resulting architecture is called a *recurrent neural network* (RNN). A common choice is to initialize the hidden state to a
default value such as

$$
z_0 = (0,0,\ldots,0)^\top.
$$

As a concrete example, consider translating sentences from English to Dutch.
Input and output sentences have variable length, and the output length may differ
from the input length. The model may also need to see the entire English
sentence before producing any Dutch words.

With an RNN we can:

1. Feed the whole English sentence word by word.
2. Then feed a special token \(\langle\text{start}\rangle\) to signal the
   beginning of the translation.

During training, the network learns that \(\langle\text{start}\rangle\)
indicates the point at which it should begin generating the translated sentence.
At each subsequent time step:

- the RNN outputs the next Dutch word,
- we feed that output word back as the next input.

The network is also trained to emit a special token
\(\langle\text{stop}\rangle\) that marks the end of the translation. We now describe the encoder--decoder RNN more explicitly. Let the English input sentence be

$$
(e_1, e_2, \ldots, e_T),
$$

and the Dutch output sentence be

$$
(d_1, d_2, \ldots, d_M).
$$

**Encoder (reads English, no outputs used).**

We start with a hidden state

$$
z_0 = \mathbf{0}.
$$

For each English word \(e_t\) we apply the recurrent update

$$
z_t = f_{\text{enc}}(z_{t-1}, e_t), \qquad t = 1,\ldots,T,
$$

where \(f_{\text{enc}}\) is the RNN cell (e.g. a simple RNN, LSTM, or GRU).
During this phase:

- *inputs*: \((z_{t-1}, e_t)\),
- *outputs*: intermediate \(z_t\); we ignore any word-level outputs.

After the last English word we keep only the final hidden state

$$
z^\ast = z_T.
$$

This \(z^\ast\) is a fixed-length vector that summarizes the whole English
sentence. It is the *encoder representation*.

**Decoder (generates Dutch, uses \(z^\ast\)).**

The decoder is another RNN (often with the same form of cell) that starts from
the encoder state. We set

$$
h_0 = z^\ast,
$$

and feed a special start token \(\langle\text{start}\rangle\) as the first input.
At step \(m = 1,2,\ldots\) we compute

$$
h_m = f_{\text{dec}}(h_{m-1}, u_m),
$$

where:

- for \(m=1\), \(u_1 = \langle\text{start}\rangle\),
- for \(m>1\), \(u_m = d_{m-1}\), the *previous* Dutch word.

From \(h_m\) the decoder produces a distribution over the next Dutch word:

$$
p(d_m \mid h_m) = \text{softmax}(W h_m + b).
$$

During training we use the true previous word \(d_{m-1}\) as input; at test time
we instead feed back the word sampled (or chosen) from \(p(d_m \mid h_m)\). The decoder continues until it outputs a special stop token
\(\langle\text{stop}\rangle\). Thus:

- *inputs to decoder*: \(z^\ast\) (via \(h_0\)) and the sequence
  \((\langle\text{start}\rangle, d_1, d_2,\ldots)\),
- *outputs from decoder*: the Dutch words
  \((d_1, d_2,\ldots,d_M,\langle\text{stop}\rangle)\).

**Autoregressive structure.**

Conditioned on the English sentence \(e_{1:T}\) (summarized by \(z^\ast\)), the
decoder defines

$$
p(d_1,\ldots,d_M \mid e_{1:T})
= \prod_{m=1}^M p\bigl(d_m \mid d_{1:m-1}, e_{1:T}\bigr),
$$

because each step \(m\) takes as input the previous hidden state \(h_{m-1}\) and
previous output word \(d_{m-1}\). This is exactly an autoregressive factorization
over the Dutch sequence, with the entire English sentence influencing each term
through \(z^\ast\) and the hidden states \(h_m\).

## Backpropagation through time

We fix a simple language-model RNN that predicts the next token at each time
step. We train RNNs with stochastic gradient descent, using gradients computed by
backpropagation and automatic differentiation, just as for standard neural
networks.

**Data, inputs, and outputs** Consider Vocabulary size \(K\), a training sequence is \(x_{1:N} = (x_1,\dots,x_N)\), with \(x_n \in \{1,\dots,K\}\). The target at step \(n\) is the next token \(t_n = x_{n+1}\) (or a special end token).

Each token is mapped to a vector \(\mathbf{x}_n\) either via a one-hot vector in \(\mathbb{R}^K\) or an embedding \(\mathbf{x}_n = \mathbf{E}\,\mathbf{e}(x_n) \in \mathbb{R}^D\),
where \(\mathbf{E} \in \mathbb{R}^{D\times K}\) is the embedding matrix.

The RNN cell has hidden state \(\mathbf{h}_n \in \mathbb{R}^H\), output logits \(\mathbf{o}_n \in \mathbb{R}^K\) and output probabilities \(\mathbf{y}_n \in \mathbb{R}^K\).

**Forward pass through time**

At each time step \(n\) the RNN computes

$$
\mathbf{h}_n = f(\mathbf{W}_{xh}\mathbf{x}_n + \mathbf{W}_{hh}\mathbf{h}_{n-1} + \mathbf{b}_h),
$$

$$
\mathbf{o}_n = \mathbf{W}_{ho}\mathbf{h}_n + \mathbf{b}_o,
$$

$$
\mathbf{y}_n = \text{softmax}(\mathbf{o}_n),
$$

with initial state \(\mathbf{h}_0\) (often \(\mathbf{0}\)). Unrolling this gives a
chain

$$
(\mathbf{x}_1,\mathbf{h}_0) \to \mathbf{h}_1 \to \mathbf{o}_1 \to \mathbf{y}_1,
\quad
(\mathbf{x}_2,\mathbf{h}_1) \to \mathbf{h}_2 \to \mathbf{o}_2 \to \mathbf{y}_2,
\quad \dots
$$

All steps share the same parameters

$$
\theta = \{\mathbf{E},\mathbf{W}_{xh},\mathbf{W}_{hh},\mathbf{W}_{ho},\mathbf{b}_h,\mathbf{b}_o\}.
$$

**Loss definition**

At step \(n\), the target is a one-hot vector \(\mathbf{t}_n \in \mathbb{R}^K\)
with components \(t_{n,k}\) and \(\sum_k t_{n,k}=1\). The cross-entropy loss at
that step is

$$
L_n = - \sum_{k=1}^K t_{n,k}\log y_{n,k}.
$$

The total loss for the sequence (over \(N'\) training steps) is

$$
L = \sum_{n=1}^{N'} L_n
  = - \sum_{n=1}^{N'} \sum_{k=1}^K t_{n,k}\log y_{n,k}.
$$

Because \(\mathbf{t}_n\) is one-hot, there is a unique index \(t_n\) with
\(t_{n,t_n}=1\) and zero for all other, so

$$
L_n = -\log y_{n,t_n}.
$$

**Backward pass**

To compute gradients we view the unrolled RNN over \(N'\) steps as one large
feed-forward network and apply standard backpropagation. Gradients at each step flow

$$
L_n \;\to\; \mathbf{y}_n \;\to\; \mathbf{o}_n \;\to\; \mathbf{h}_n
\;\to\; \mathbf{h}_{n-1} \;\to\; \theta.
$$

- The recurrent connection \(\mathbf{h}_{n-1} \to \mathbf{h}_n\) causes the
  gradient at time \(n\) to contribute to the gradient at all earlier times. Since the same parameters \(\theta\) are reused at every time step, the total
  gradient is the sum of contributions from all steps:

$$
\frac{\partial L}{\partial \theta}
= \sum_{n=1}^{N'} \frac{\partial L}{\partial \theta}\Big|_{\text{via step }n}.
$$

Running this backward pass over the unrolled network is called
*backpropagation through time (BPTT)*. Conceptually it is straightforward,
but in practice very long sequences cause training difficulties: gradients can
either vanish or explode, just as in very deep feed-forward networks.

### Long range dependencies
  
Standard RNNs also struggle with long-range dependencies. Natural language
often contains concepts introduced early in a passage that strongly influence
words much later. In the encoder–decoder architecture described earlier, the
entire meaning of the English sentence must be stored in a single fixed-length
hidden vector \(z^\ast\). As sequences grow longer, compressing all relevant
information into \(z^\ast\) becomes harder. This is called a *bottleneck*
problem: an arbitrarily long input sequence must be summarized into one hidden
vector before the network can start producing the output translation.

To address both vanishing/exploding gradients and limited long-range memory,
we can change the recurrent cell to include additional pathways that let signals
bypass many intermediate computations. This helps information persist over more
time steps. The most prominent examples are *long short-term memory*
(LSTM) networks and *gated recurrent
units* (GRU). These architectures improve performance over
standard RNNs, but they still have restricted ability to capture very long-range
dependencies. Their more complex cells also make them slower to train. All recurrent models, including LSTMs and GRUs, share two structural
limitations:

- The length of the signal path between distant time steps grows
  linearly with the sequence length. To see this, look at the unrolled RNN:

$$
 (\mathbf{x}_1,\mathbf{h}_0) \to \mathbf{h}_1 \to \mathbf{h}_2 \to \cdots \to \mathbf{h}_N \to \mathbf{y}_N.
$$

  Any influence from time step \(i\) on time step \(j>i\) must pass through the
  chain

$$
 \mathbf{x}_i \to \mathbf{h}_i \to \mathbf{h}_{i+1} \to \cdots \to \mathbf{h}_j \to \mathbf{y}_j.
$$

  This path contains \((j-i)\) recurrent transitions
  \(\mathbf{h}_{t-1}\to\mathbf{h}_t\) (plus a constant number of input/output
  edges), so its length is proportional to \(j-i\). For two positions that are
  \(L\) steps apart, the signal must traverse \(O(L)\) nonlinear transformations.
  
- Computation within a single sequence is inherently sequential, so
  different time steps cannot be processed in parallel.

As a result, RNNs cannot exploit modern highly parallel hardware (such as GPUs)
efficiently. These limitations motivate replacing RNNs with transformer
architectures.


## References
- Bishop, C. M., & Bishop, H. (2023). Transformers. In Deep Learning: Foundations and Concepts (pp. 357-406). Cham: Springer International Publishing.
