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
inputs, so the network is trained to ``fill in the blank''. In the
*skip-gram* model, the roles are reversed: the centre word is the input and
the context words are the targets.

This training can be viewed as *self-supervised* learning. The data is a
large corpus of unlabelled text, from which many small windows of word
sequences are sampled at random. The labels come from the text itself by
``masking'' the word whose value the network should predict. After training, the embedding matrix \(\mathbf{E}\) is obtained from the network
weights: it is the transpose of the second-layer weight matrix for the CBOW
model, and the first-layer weight matrix for the skip-gram model. Here is a compact derivation for both CBOW and skip-gram.

**Why the embedding matrix equals those weight matrices**

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

1. **First layer (input \(\to\) hidden).**

   Weight matrix \(\mathbf{W}^{(1)} \in \mathbb{R}^{D \times K}\). For one context word \(w\),

$$
\mathbf{h}(w) = \mathbf{W}^{(1)} \mathbf{x}(w).
$$

   For a window of \(M\) context words \(w_1,\dots,w_M\), CBOW uses the average

$$
\mathbf{h} = \frac{1}{M} \sum_{i=1}^M \mathbf{W}^{(1)} \mathbf{x}(w_i).
$$

2. **Second layer (hidden \(\to\) output).**

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

So in CBOW the “learned embeddings” are not something extra: they are exactly
the parameters of the output layer that connect the context representation
\(\mathbf{h}\) to each possible target word.

#### Skip-gram

We have a corpus \(w_1, w_2, \dots, w_T\). At each position \(t\) we treat \(w_t\) as
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

using all such pairs or a random subset.

**Architecture**

1. **First layer (input \(\to\) hidden).**

   Weight matrix \(\mathbf{W}^{(1)} \in \mathbb{R}^{D \times K}\).

   Input is the one-hot vector \(\mathbf{x}(w_t)\) with shape \(K \times 1\) for the centre word \(w_t\).
   The hidden vector for the input (center) word is given by:

$$
\mathbf{h} = \mathbf{W}^{(1)} \mathbf{x}(w_t).
$$

   Because \(\mathbf{x}(w_t)\) has a single 1 and all zeros, \(\mathbf{h}\) is the \(t\)-th column
   of \(\mathbf{W}^{(1)}\) and has the shape \(D \times 1\). This give a \(D\) dimensional representation vector for the input word.

2. **Second layer (hidden \(\to\) output).**

   Weight matrix \(\mathbf{W}^{(2)} \in \mathbb{R}^{K \times D}\).

   For all context position \(j\) we use the same hidden vector
   \(\mathbf{h} \in \mathbb{R}^D\) and the same \(\mathbf{W}^{(2)}\). Let
   \(\mathbf{w}^{(2)}_k{}^{\!\top}\) be the \(k\)th row of \(\mathbf{W}^{(2)}\),
   so \(\mathbf{w}^{(2)}_k \in \mathbb{R}^D\) is a feature vector for
   candidate word \(k\). The logit (a scalar) is

$$
z_k = \mathbf{w}^{(2)}_k{}^{\!\top} \mathbf{h},
$$

   We compute their dot product, so if these two vectors point in a similar direction (high similarity),
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

Even though \(j\) is ignored:

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
more often than unrelated words. For example, the words ``city'' and
``capital'' appear more often as context for target words such as ``Paris'' or
``London'' than for ``orange'' or ``polynomial''. The network can then more
easily predict the missing words if ``Paris'' and ``London'' are mapped to
nearby embedding vectors.

The learned embedding space often has richer semantic structure than simple
closeness of related words, and it supports simple vector arithmetic. For
example, the relation ``Paris is to France as Rome is to Italy'' can be
expressed in terms of embedding vectors. Writing \(\mathbf{v}(\text{word})\) for
the embedding of *word*, we find

$$
\mathbf{v}(\text{Paris}) - \mathbf{v}(\text{France})
+ \mathbf{v}(\text{Italy}) \simeq \mathbf{v}(\text{Rome}) .
\tag{12.27}
$$

Word embeddings were first developed as stand-alone tools for natural language
processing. Today they are more often used as pre-processing steps for deep
neural networks, and can be viewed as the first layer of such a network. The
embedding matrix may be fixed, using some standard pre-trained embeddings, or
treated as an adaptive layer that is learned during end-to-end training of the
whole system. In the adaptive case, the embedding layer can be initialized with
random weights or with a standard pre-trained embedding matrix.

