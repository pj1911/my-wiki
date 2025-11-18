# Transformers - Introduction

## Introduction
Transformers are a major breakthrough in deep learning. They use attention as a way for the network to give different weights to different inputs, where those weights are determined by the inputs themselves. This lets transformers naturally capture useful patterns in sequences and other kinds of data. They’re called transformers because they take a set of vectors in one space and turn them into a new set of vectors (with the same size) in another space. This new space is designed such that it holds a richer internal representation that makes it easier to solve downstream tasks.

One big strength of transformers is that transfer learning works really well with them. We can first train a transformer on a huge amount of data, then fine-tune that same model for many different tasks. When a large model like this can be adapted to lots of tasks, we call it a foundation model.

Transformers can also be trained in a self-supervised way on unlabeled data, which is perfect for language, since there’s so much text available on the internet and elsewhere. The scaling hypothesis says that if we just make the model bigger (more parameters) and train it on more data, we get much better performance even without changing the architecture. Transformers also run very efficiently on GPUs, which support massive parallel processing, so we can train huge language models with around a trillion parameters in a reasonable time. Now we will go through the different components of the Transformer architecture:

## Attention
The core idea behind transformers is attention. It was first created to improve RNNs for machine translation but later, it was shown that we could remove recurrence entirely and rely only on attention, getting much better results. Today, attention-based transformers have largely replaced RNNs in almost all applications. For example, consider the following sentences:

- The baseball player gripped the bat.
- A small animal that hangs upside-down might be a bat.

Here the word “bat” has different meanings in the two sentences. However, this can be detected only by looking at the context provided by the other words in the sequence. We also see that some words are more important than others in determining the interpretation of “bat.” In the first sentence, the words “baseball player” and “gripped” strongly indicate that “bat” refers to a piece of sports equipment, whereas in the second sentence, the words “small animal” and “hangs upside-down” indicate that “bat” refers to a flying mammal. Thus, to determine the appropriate interpretation of “bat,” a neural network processing such a sentence should attend to—i.e., rely more heavily on these specific words from the rest of the sequence. 

In a standard neural network, each input affects the output based on its weight, and once the network is trained, these weights stay fixed. But based on the above examples, we want the model to focus on different words in different positions for each new input. Attention makes this possible by using weights that change depending on the specific input data.

In natural language processing we will see that word embeddings map each word to a vector in an embedding space. These vectors are then used as inputs to neural networks. The embeddings capture basic meaning: words with similar meanings end up close together in this space. A key point is that each word always maps to the same vector (for example, “bat” always has one fixed embedding even when the context is different, like in our example above). Meanwhile, a transformer can be seen as a more powerful kind of embedding. It maps each word’s vector to a new vector that depends on the other words in the sequence. That means the word “bat” can end up in different places depending on the sentence: near “baseball” or near “animal”.

### Transformer processing

In a transformer, the input is a set of vectors \(\{x_n\}\) of dimensionality \(D\),
for \(n = 1, \dots, N\). Each vector is called a *token*. A token might
correspond to a word in a sentence or a patch in an image.

The individual components \(x_{ni}\) of each token are called *features*. A key advantage of transformers is that we do not need to design a different
neural network architecture for each data type. Instead, we simply convert the
different kinds of data into a shared set of tokens and feed them into the same
model.

### Notations
We stack the token vectors for one sequence into a data matrix \(\mathbf{X}\) of shape \(\mathbf{N} \times \mathbf{D}\), where each row (\(x_n^T\)) is a token with \(\mathbf{D}\) columns or features. In real tasks, we have many such sequences (for example, many text passages, 
with each word represented as one token). 

A fundamental building block of the transformer is the *transformer layer*, a function that takes \(\mathbf{X}\) as input 
and outputs a new matrix \(\tilde{\mathbf{X}}\) of the same size:

$$
\tilde{\mathbf{X}} = \mathrm{TransformerLayer}[\mathbf{X}] \, .
$$

By stacking several transformer layers, we obtain a deep network that can learn 
rich internal representations. Each transformer layer has its own weights and 
biases, which are learned using gradient descent with an appropriate cost function.

A single transformer layer has two stages. The first stage implements the attention mechanism, which, for each feature column, forms a weighted sum over all tokens (rows), thereby mixing information between the token vectors. The second stage then acts on each row 
independently and further transforms the features within each token vector.

### Attention coefficients
Suppose we have a set of input token vectors (rows)

$$
\mathbf{x}_1, \dots, \mathbf{x}_N
$$

and we want to map them to a new set of output vectors

$$
\mathbf{y}_1, \dots, \mathbf{y}_N
$$

in a new embedding space that captures richer semantic structure.

For any particular output vector \(\mathbf{y}_n\), we want it to depend not only on
its corresponding input \(\mathbf{x}_n\) but on all input token vectors (rows)
\(\mathbf{x}_1, \dots, \mathbf{x}_N\). A simple way to achieve this is to define
\(\mathbf{y}_n\) as a weighted sum of the inputs:

$$
\mathbf{y}_n = \sum_{m=1}^{N} a_{nm}\,\mathbf{x}_m,
$$

where the coefficients \(a_{nm}\) are called *attention weights*.

We require these coefficients to satisfy

$$
a_{nm} \ge 0 \quad \text{for all } m
$$

and

$$
\sum_{m=1}^{N} a_{nm} = 1.
$$

These constraints ensure that the weights form a partition of unity, so that
each coefficient lies in the range \(0 \le a_{nm} \le 1\). Thus, each output
vector \(\mathbf{y}_n\) is a convex combination (a weighted average) of the input
vectors, with some inputs receiving larger weights than others as we wanted.

Note that we have a different set of coefficients \(\{a_{n1}, \dots, a_{nN}\}\)
for each output index \(n\), and the above constraints apply separately for each
\(n\). The coefficients \(a_{nm}\) depend on the input data, later we will see how
we compute them in practice.

### Self attention
We wish to determine the coefficients \(a_{nm}\) used in

$$
\mathbf{y}_n = \sum_{m=1}^{N} a_{nm}\,\mathbf{x}_m.
$$

**Query, Key and Value analogy.**  
In information retrieval (e.g. a movie streaming service), each movie is
described by an attribute vector called a *key*, while the movie file
itself is a *value*.  
A user specifies their preferences as a *query* vector.  
The system compares the query with all keys, finds the best match, and returns
the corresponding value.  
Focusing on a single best-matching movie would be called *hard attention*.

In transformers we use *soft attention*: instead of returning a single
value, we compute continuous weights that tell us how strongly each value
should influence the output. This keeps the whole mapping differentiable, so it
can be trained by gradient descent.

**Applying this to tokens.**  
For each input token we start from its embedding \(\mathbf{x}_n\), but we use
*three* conceptually different copies of it (in this simple version we set
\(\mathbf{q}_n = \mathbf{k}_n = \mathbf{v}_n = \mathbf{x}_n\) for all \(n\)).

- **Value** \(\mathbf{v}_n\) *(movie file)*: the actual content to return or mix into the output.
- **Key** \(\mathbf{k}_n\) *(movie's attribute profile)*: a summary describing that movie (genre, actors, length) used for matching.
- **Query** \(\mathbf{q}_n\) *(user's wish list of attributes)*: what the output position is looking for and this is compared against all keys.

To decide how much the output at position \(n\) should attend to token \(m\), we
measure the similarity between the corresponding query and key vectors.  A
simple similarity measure is the dot product

$$
\mathbf{x}_n^\top \mathbf{x}_m.
$$

**Attention weights via softmax.**  
To enforce the constraints

$$
a_{nm} \ge 0, \qquad
\sum_{m=1}^{N} a_{nm} = 1 \quad \text{for each fixed } n,
$$

we define the attention weights by a softmax over \(m\):

$$
a_{nm}
= \frac{\exp\big(\mathbf{x}_n^\top \mathbf{x}_m\big)}
       {\sum_{m'=1}^{N} \exp\big(\mathbf{x}_n^\top \mathbf{x}_{m'}\big)}.
$$

Thus, for each \(n\), the row \((a_{n1},\dots,a_{nN})\) forms a set of
non-negative coefficients that sum to one, assigning larger weights to inputs
whose keys are more similar to the query.

**Matrix form and self-attention.**  
Let \(\mathbf{X} \in \mathbb{R}^{N \times D}\) be the input matrix whose \(n\)-th
row is \(\mathbf{x}_n\), and let \(\mathbf{Y} \in \mathbb{R}^{N \times D}\) be the
output matrix whose \(n\)-th row is \(\mathbf{y}_n\).  
We first compute all pairwise dot products:

$$
\mathbf{L} = \mathbf{X}\mathbf{X}^\top \in \mathbb{R}^{N \times N}.
$$

We then apply the softmax row-wise to obtain the attention matrix

$$
\mathbf{A} = \mathrm{Softmax}[\mathbf{X}\mathbf{X}^\top],
$$

and finally compute the outputs as

$$
\mathbf{Y} = \mathbf{A}\mathbf{X}
           = \mathrm{Softmax}[\mathbf{X}\mathbf{X}^\top]\,\mathbf{X}.
$$

Because the queries, keys, and values are all derived from the same sequence
\(\mathbf{X}\), this mechanism is known as *self-attention*, and since the
similarity is given by a dot product, it is specifically *dot-product
self-attention*.

### Network parameters

We currently have two problems:
- Vanilla self-attention contains no trainable parameters
- Treat all feature values within a token equally in determining the attention coefficients.

Introducing a single projection (learnable weight matrix) \(\mathbf{U}\in\mathbb{R}^{D\times D}\),

$$
\tilde{\mathbf{X}}=\mathbf{X}\mathbf{U},\qquad
\mathbf{Y}=\mathrm{Softmax}\!\big[\mathbf{X}\mathbf{U}\mathbf{U}^\top\mathbf{X}^\top\big]\;\mathbf{X}\mathbf{U},
$$

adds learnability but yields a symmetric score matrix
\(\mathbf{X}\mathbf{U}\mathbf{U}^\top\mathbf{X}^\top\) and ties the value and
similarity parameters.

To obtain a more flexible, asymmetric mechanism (for example, bat should be strongly associated with animal but animal should not be strongly associated with bat as there are different kinds of animals), therefore, we use independent projections
for queries, keys, and values:

$$
\mathbf{Q}=\mathbf{X}\mathbf{W}^{(q)},\quad
\mathbf{K}=\mathbf{X}\mathbf{W}^{(k)},\quad
\mathbf{V}=\mathbf{X}\mathbf{W}^{(v)},
$$

with linear trasnformations \(\mathbf{W}^{(q)},\mathbf{W}^{(k)}\in\mathbb{R}^{D\times D_k}\) and
\(\mathbf{W}^{(v)}\in\mathbb{R}^{D\times D_v}\) (typically \(D_k=D\), \(D_v=D\) as this helps to stack multiple Transformer layers on top of each other).
The resulting dot-product self-attention is

$$
\mathbf{Y}=\mathrm{Softmax}\!\big[\mathbf{Q}\mathbf{K}^\top\big]\;\mathbf{V},
$$

with \(\mathbf{Q}\mathbf{K}^{T}\in\mathbb{R}^{N\times N}\) and \(\mathbf{Y}\in\mathbb{R}^{N\times D_v}\), which is trainable, reweighs features, and supports asymmetric token
relationships.

**Bias absorption.**  
Add a column of ones to the data and a row for biases to the weights, so that

$$
XW + \mathbf{1} b^\top
= \underbrace{\big[\,X\;\;\mathbf{1}\,\big]}_{X_{\text{aug}}}
\underbrace{\begin{bmatrix} W \\ b^\top \end{bmatrix}}_{W_{\text{aug}}}.
$$

Hence, biases can be treated as implicit via augmentation.

**Fixed vs data dependent weights - Simple NN vs Transformer.**  
For a simple NN let a single input vector be

$$
\mathbf{x}\in\mathbb{R}^{D_{\text{in}}}
\quad\text{and the layer have } D_{\text{out}} \text{ output units.}
$$

The layer has a weight matrix \(W\in\mathbb{R}^{D_{\text{in}}\times D_{\text{out}}}\).
The output is

$$
\mathbf{y}=\mathbf{x}W \quad(\text{or } \mathbf{y}=\mathbf{x}W).
$$

Component-wise, for each output unit \(n\in\{1,\dots,D_{\text{out}}\}\),

$$
y_n=\sum_{m=1}^{D_{\text{in}}} x_m\,W_{m n}.
$$

So each output unit is a *weighted sum* of *all* input features. But these weights \(W_{mn}\) are fix for all inputs.

**Transformers data dependent weights.**  
In the standard layer above, once training is done, the weights \(W_{mn}\) are
*fixed*. For any new input \(\mathbf{x}\), the contribution of input feature
\(m\) to output \(n\) is always \(x_m\,W_{mn}\).

In *attention*, by contrast, the mixing coefficients are computed from the
*current input*. With queries, keys, and values
\(\,Q=XW^{(q)},\,K=XW^{(k)},\,V=XW^{(v)}\). Therefore, Q,K and V will be different for different inputs.

$$
Y = \underbrace{\mathrm{Softmax}\!\big(QK^\top\big)}_{\displaystyle A(X)}
\,V,
\qquad
\mathbf{y}_n=\sum_{m=1}^{N} a_{nm}(X)\,\mathbf{v}_m,
$$

and the “weights” \(a_{nm}(X)\) depend on the present data (via a softmax over
dot products).Thus, the contribution from token \(m\) to output \(n\) can be nearly zero for one input and large for another a behavior that a standard fixed-weight layer cannot achieve.

### Scaled self attention

**Issue.** Softmax gradients shrink when its inputs (logits) are large in
magnitude (saturation). In dot-product attention the logits are
\(\ell_{ij}=\mathbf{q}_i^\top\mathbf{k}_j\), which can grow with vector
dimension.

**Why do they grow?** Assume (as a scale reference) that query/key
components are independent with mean \(0\) and variance \(1\):
\(\mathbf{q}=(q_1,\dots,q_{D_k})\), \(\mathbf{k}=(k_1,\dots,k_{D_k})\).
Then

$$
\mathbf{q}^\top\mathbf{k}=\sum_{t=1}^{D_k} q_t k_t,
\qquad
\mathbb{E}[q_t k_t]=0,\quad
\mathrm{Var}(q_t k_t)=\mathbb{E}[q_t^2]\mathbb{E}[k_t^2]=1\cdot 1=1,
$$

so by independence,

$$
\mathrm{Var}(\mathbf{q}^\top\mathbf{k})
=\sum_{t=1}^{D_k}\mathrm{Var}(q_t k_t)=D_k,
$$

and the typical magnitude (std. dev.) is \(\sqrt{D_k}\). Larger \(D_k\) therefore
pushes logits to larger scales, sharpening the softmax and shrinking gradients.

**Fix.** Normalize logits by their standard deviation:

$$
\tilde{\ell}_{ij}=\frac{\mathbf{q}_i^\top\mathbf{k}_j}{\sqrt{D_k}},
$$

which makes \(\mathrm{Var}(\tilde{\ell}_{ij})\approx 1\) under the reference
assumption, keeping softmax in a stable range.

**Result.** The attention layer is

$$
\mathbf{Y} = Attention(\mathbf{Q},\mathbf{K},\mathbf{V})
=\mathrm{Softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{D_k}}\right)\mathbf{V}.
$$

This is *scaled* dot-product self-attention. (Even when the independence/variance
assumptions are only approximate, this scaling acts like a principled temperature
that stabilizes training.)

### Multi head attention
A single head can average out distinct patterns. Therefore, we use \(H\) parallel heads with separate
parameters to attend to different patterns for example in NLP it can be tenses, vocabulary, etc.

**Setup.** Let \(X\!\in\!\mathbb{R}^{N\times D}\).
Each head \(h\in\{1,\dots,H\}\) has its own:

$$
Q_h = X W^{(q)}_h,\qquad
K_h = X W^{(k)}_h,\qquad
V_h = X W^{(v)}_h,
$$

with \(W^{(q)}_h,W^{(k)}_h\!\in\!\mathbb{R}^{D\times D_k}\) and
\(W^{(v)}_h\!\in\!\mathbb{R}^{D\times D_v}\).

**Per-head attention (scaled).**

$$
\mathbf{H}_h \;=\; Attention(\mathbf{Q_h}, \mathbf{K_h},\mathbf{V_h}) \;=\; \mathrm{Softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{D_k}}\right) V_h
\;\in\;\mathbb{R}^{N\times D_v}.
$$

**Combine heads.** Concatenate along features to get shape \((N \times H D_v)\) and project:

$$
Y(X) \;=\; \mathrm{Concat}\,[H_1,\dots,H_H]\; W^{(o)},\qquad
W^{(o)} \in \mathbb{R}^{H D_v \times D }\text{ is a trainable linear matrix: },
$$

so \(Y\in\mathbb{R}^{N\times D}\) matches the input width.
A common choice is \(D_k=D_v=D/H\), making the concatenated matrix \(N\times D\).

**Redundancy observed.**
It is due to the reparameterization on the value path.
We know for each head,

$$
H_h=\mathrm{Softmax}\!\Big(\tfrac{Q_h K_h^\top}{\sqrt{D_k}}\Big)\;V_h,
\qquad
V_h = X W^{(v)}_h,
$$

and the final combine is

$$
Y=\mathrm{Concat}[H_1,\dots,H_H]\;W^{(o)}.
$$

We can write \(W^{(o)}=\big[(W^{(o)}_1)^\top\;\cdots\;(W^{(o)}_H)^\top\big]^\top\).
Then

$$
Y=\sum_{h=1}^H H_h\,W^{(o)}_h
  =\sum_{h=1}^H \mathrm{Softmax}\!\Big(\tfrac{Q_h K_h^\top}{\sqrt{D_k}}\Big)\;\underbrace{(X W^{(v)}_h W^{(o)}_h)}_{X\,\tilde W^{(v)}_h}.
$$

Then we can define \(\tilde W^{(v)}_h := W^{(v)}_h W^{(o)}_h\). The layer becomes

$$
Y=\sum_{h=1}^H \mathrm{Softmax}\!\Big(\tfrac{Q_h K_h^\top}{\sqrt{D_k}}\Big)\;X\,\tilde W^{(v)}_h,
$$

with no explicit \(W^{(o)}\).

Therefore the two consecutive linear maps on \(V\) (first \(W^{(v)}_h\),
then the head’s block \(W^{(o)}_h\)) can always be merged into a single matrix
\(\tilde W^{(v)}_h\). Since the attention weights use only \(Q\) and \(K\), the value path is purely
linear per head: \(V_h W^{(o)}_h = X\,W^{(v)}_h W^{(o)}_h\).
Thus we can collapse the two matrices into one
\(\tilde W^{(v)}_h := W^{(v)}_h W^{(o)}_h\).
This gives two equivalent parameterizations:

$$
\text{(separate)}\;\; (W^{(v)}_h,\,W^{(o)}_h)
\quad\longleftrightarrow\quad
\text{(collapsed)}\;\; \tilde W^{(v)}_h.
$$

**This non-uniqueness is the redundancy.** we keep two matrices even
though one combined matrix suffices to represent exactly the same function.

Why keep \(W^{(o)}\) in practice? It standardizes the output width \(D\),
keeps per-head value sizes \(D_v\) small, and matches common implementations, 
but representationally, only the product \(W^{(v)}_h W^{(o)}_h\) matters.

### Transformer layers

The input tokens have the shape:  \((X\in\mathbb{R}^{N\times D})\) (where rows are the tokens).

**Multi-head attention (MHA).** For heads \((h=1,\dots,H)\),

$$
Q_h=XW^{(q)}_h,\quad K_h=XW^{(k)}_h,\quad V_h=XW^{(v)}_h,
$$

$$
H_h=\mathrm{Softmax}\!\Big(\tfrac{Q_h K_h^\top}{\sqrt{D_k}}\Big)\,V_h\in\mathbb{R}^{N\times D_v}.
$$

Concatenate and project:

$$
\mathrm{MHA}(X)=\mathrm{Concat}[H_1,\dots,H_H]\;W^{(o)},\qquad
W^{(o)}\in\mathbb{R}^{H D_v\times D}.
$$

The output from MHA layer has the same shape as its input of \((N \times D)\) enabling residuals. MHA gives data-dependent mixing between tokens and helps
learn different relations (e.g., syntax vs. semantics) in parallel.

**Residual + LayerNorm (two variants).**
Residuals preserve the input signal and enable deep stacks. In addition to this, pre/post
norm improve optimization stability (pre-norm is common for very deep models).

*Post-norm:*

$$
Z=\mathrm{LayerNorm}\big(\mathrm{MHA}(X)+X\big).
$$

*Pre-norm:*

$$
Z=X+\mathrm{MHA}(\mathrm{LayerNorm}(X)).
$$

**Position-wise MLP (shared across tokens).**
MHA outputs lie in the span of inputs due to linear mixing, the MLP adds nonlinearity and feature-wise transformation per token, boosting expressiveness. A two-layer example with activation \(\phi\) can be denoted as:

$$
\mathrm{MLP}(U)=\phi(UW_1+b_1)\,W_2+b_2,\quad
W_1\in\mathbb{R}^{D\times D_{\mathrm{ff}}},\;\;W_2\in\mathbb{R}^{D_{\mathrm{ff}}\times D}.
$$

**Block output (two variants).**

*Post-norm:*

$$
\tilde X=\mathrm{LayerNorm}\big(\mathrm{MLP}(Z)+Z\big).
$$

*Pre-norm:*

$$
\tilde X=Z+\mathrm{MLP}(\mathrm{LayerNorm}(Z)).
$$

Layer normalization is generally used in both sublayers (MHA and MLP) to normalize per token to reduce covariate shift and to keep activations in a
well-scaled regime to keep the training process steady.

**Stacking.** Repeat the block \(L\) times to form a
deep transformer. Note that all mappings preserve shape \(N\times D\), enabling residuals.

### Computational complexity

**Setup.** Input \(X\in\mathbb{R}^{N\times D}\) (rows = tokens with \(D\) features).
A transformer block outputs the same shape \(N\times D\) as its input.
We use \(H\) heads and for each head key/query/value widths (features) are \(d_k,d_k,d_v\) (typically
\(d_k=d_v=D/H\)).

**Baseline: fully connected on flattened sequence**  
Flatten \(X\) to a vector in \(\mathbb{R}^{ND}\) and map to \(\mathbb{R}^{ND}\) with weight matrix
\(W\in\mathbb{R}^{ND\times ND}\).

$$
\text{parameters}= \text{elements in the weight matrix = }N^2D^2,\qquad
\text{ FLOPs }\approx  2\,N^2D^2.
$$

Where, FLOPs mean Floating-point Operations.

**Multi-head self-attention (MHA)**

**1) Linear projections to \(Q,K,V\).**  
For each head \(h\):

$$
Q_h=XW^{(q)}_h,\quad K_h=XW^{(k)}_h,\quad V_h=XW^{(v)}_h.
$$

with \(W^{(q)}_h,W^{(k)}_h\in\mathbb{R}^{D\times d_k}\), \(W^{(v)}_h\in\mathbb{R}^{D\times d_v}\).

$$
\begin{aligned}
\text{Shapes: }& Q_h,K_h\in\mathbb{R}^{N\times d_k},\; V_h\in\mathbb{R}^{N\times d_v}.\\

\text{FLOPs: }& \underbrace{N D d_k}_{XW^{(q)}_h}+\underbrace{N D d_k}_{XW^{(k)}_h}
+\underbrace{N D d_v}_{XW^{(v)}_h}\;\text{ per head}.\\

&\Rightarrow\;\text{Total FLOPs (all \(H\) heads)}=N D\,(2H d_k + H d_v) = 3ND^2.
\end{aligned}
$$

$$
\text{parameters (projections)}=D(2H d_k + H d_v) = 3D^2.
$$

**2) Attention scores (scaled dot products).**  
For each head:

$$
S_h=\frac{Q_h K_h^\top}{\sqrt{d_k}}\in\mathbb{R}^{N\times N}.
$$

$$
\text{FLOPs: }N^2 d_k \text{ (matrix multiply)}\quad (\text{the divide by }\sqrt{d_k}\text{ is }O(N^2)).
$$

Softmax over rows:

$$
A_h=\mathrm{Softmax}(S_h)\in\mathbb{R}^{N\times N},\qquad
\text{FLOPs: }O(N^2).
$$

Total FLOPs (all heads): \(H N^2 d_k\) for \(QK^\top\) and \(O(H N^2)\) for softmax and parameters = 0.

**3) Value mixing.**  
For each head:

$$
H_h=A_h V_h\in\mathbb{R}^{N\times d_v},\qquad
\text{FLOPs: }N^2 d_v = \frac{N^2D}{H}
 \text{ (matrix multiply)}, \qquad \text{parameters}=0
$$

**4) Concatenate and output projection.**  
Concatenate \(H_h\) along features:

$$
H=\mathrm{Concat}[H_1,\dots,H_H]\in\mathbb{R}^{N\times (H d_v)}.
$$

Project to width \(D\):

$$
Y_{\text{attn}}=H W^{(o)},\quad W^{(o)}\in\mathbb{R}^{H d_v\times D}.
$$

$$
\text{FLOPs: }N\,(H d_v)\,D,\qquad
\text{parameters (output proj)}=(H d_v)D.
$$

**Common setting \(d_k=d_v=D/H\).**  
Then

$$
\begin{aligned}
\text{Parameters (MHA)} &= D(2H\cdot \tfrac{D}{H} + H\cdot \tfrac{D}{H}) + 0 + 0 + (H\tfrac{D}{H})D
= 4D^2,\\
\text{FLOPs (MHA)} &=
\underbrace{N D (2H d_k + H d_v)}_{=\,3ND^2}
+\underbrace{H N^2 d_k}_{=\,N^2 D}
+\underbrace{H N^2 d_v}_{=\,N^2 D}
+\underbrace{N(H d_v)D}_{=\,N D^2}\\
&= \boxed{2N^2 D + 4 N D^2}\quad(\text{softmax adds }O(N^2)).
\end{aligned}
$$

**Residual adds**  
\(X+Y_{\text{attn}}\): elementwise add, \(\;\text{FLOPs}=N D, \qquad \#\text{parmas} = 0 \)

**Layer normalization**  
Given an input token (row) \(x \in \mathbb{R}^D\), LayerNorm computes

$$
\mu = \frac{1}{D} \sum_{i=1}^D x_i, 
\qquad
\sigma^2 = \frac{1}{D} \sum_{i=1}^D (x_i - \mu)^2,
$$

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}},
\qquad \text{final output is given by: }
y_i = \gamma_i \hat{x}_i + \beta_i,
\quad i = 1,\dots,D,
$$

where \(\gamma, \beta \in \mathbb{R}^D\) are learnable scale and bias.

For a batch of \(N\) tokens (matrix \(X \in \mathbb{R}^{N \times D}\)), FLOPs (in forward pass):

$$
\text{FLOPs} \approx c \, N D
$$

for a small constant \(c\) (mean/var + normalize + scale/shift).

Parameters per LayerNorm:

$$
\text{parameters} = 2D \quad (\gamma,\beta).
$$

**Position-wise MLP (shared across tokens)**  
Two-layer MLP with hidden width \(D_{\text{ff}}\):

$$
\mathrm{MLP}(U)=\phi(UW_1+b_1)\,W_2+b_2,\quad
W_1\in\mathbb{R}^{D\times D_{\text{ff}}},\;W_2\in\mathbb{R}^{D_{\text{ff}}\times D}.
$$

$$
\text{parameters for }W_1, W_2, b_1, b_2 = D D_{\text{ff}} + D_{\text{ff}} D + D_{\text{ff}} + D
\approx 2 D D_{\text{ff}}.
$$

**FLOPs.** The two matrix multiplications dominate:
\(\;UW_1: (N\times D)(D\times D_{\text{ff}})\) and \((\cdot)W_2: (N\times D_{\text{ff}})(D_{\text{ff}}\times D)\),
each costing \(\approx  N D D_{\text{ff}}\) FLOPs, so in total
\(\text{FLOPs} \approx 2 N D D_{\text{ff}}\).

Common choice \(D_{\text{ff}}=cD\) (e.g.\(c{=}4\)) gives \(\text{parameters}\approx 2c D^2\)
and \(\text{FLOPs}=2c N D^2\).

**Block totals (one transformer block, pre-/post-norm similar)**  
Ignoring small \(ND\) terms from residuals/LayerNorm:

$$
\boxed{
\text{FLOPs} \;\approx\; \underbrace{2 N^2 D}_{\text{attention mixes}}
\;+\; \underbrace{4 N D^2}_{\text{QKV+out proj}}
\;+\; \underbrace{2c N D^2}_{\text{MLP}}
}
$$

$$
\text{FLOPs} \;\approx\; 2 N^2 D \;+\; (4+2c)\,N D^2.
$$

$$
\boxed{
\text{parameters} \;\approx\; \underbrace{4 D^2}_{\text{MHA}}
\;+\; \underbrace{2c D^2}_{\text{MLP}}
\;+\; \underbrace{4D}_{\text{two LayerNorms}}
}
$$

**When does which term dominate?**  
Attention dominates for long sequences (\(N\gg D\)) since it scales as \(N^2 D\).
The MLP dominates for wide models (\(D\gg N\)) since it scales as \(N D^2\).
Compared to a dense \(\mathbb{R}^{ND}\!\to\!\mathbb{R}^{ND}\) layer
(\(\Theta(N^2D^2)\) params/FLOPs), a transformer block is vastly more efficient.

### Positional encoding

**Why we need it.**  
Transformers have a really cool property, since it shares \((W_h^{(q)},W_h^{(k)},W_h^{(v)})\) across input tokens and applies the
same computations to every row of \(X\in\mathbb{R}^{N\times D}\).  
This makes permuting the input rows results in the same permutation of the rows of the output matrix (permutation equivariance). Since parameters are shared across inputs, it gives two major benefits to the network, firstly it makes the computation parallel, and secondly it makes long range dependencies just as effective as the short range ones. This is because the same weight matrices (for attention, feed-forward layers, etc.) are shared across all tokens in the sequence, the model doesn’t need different parameters for each position or word. This means every token can be run through the same computations at the same time, letting GPUs/TPUs process all tokens in parallel instead of one after another. But for sequences (language, etc.), order matters, so we need to inject
positional information into the data. Since we want to keep the two nice properties of our attention layers discussed above, we’d rather encode token order directly in the input representations, instead of baking it into the network architecture itself.

**Additive positional encoding.**  
Associate each position \(n\) with a vector \(r_n\in\mathbb{R}^{D}\) as each token has \(D\) dimensional features and add it to the
token embedding \(x_n\):

$$
\tilde x_n = x_n + r_n \quad(=\text{row } n \text{ of } \tilde X).
$$

This might seem counter-intuitive, as this might mess up the input vector, but in reality this works really well. As in high-dimensional spaces, two unrelated vectors are almost orthogonal, so the model can keep token identity and position information relatively separate even when they’re added. Residual connections across layers help preserve this position information as it flows through the network. And because the layers are mostly linear, using addition of token and position embeddings behaves a lot like concatenating them and then applying a linear layer (add then linear is a special case of concatenation then linear where the bigger weight matrix is a concatenation of two same smaller weight matrices). This also preserves the model’s width \(D\) (concatenation would increase cost).

**Example**  
The simplest example for this is \(r_n=\{1,2,3,\cdots\}\). But in this case the magnitude can get very high and start corrupting the input. In addition, it may not generalize well to new input sequences that are longer than samples in training dataset.

**Design goals.**  
A good positional code should be: (i) unique per position, (ii) bounded,
(iii) generalize to longer sequences, and (iv) support relative offsets.
Therefore, a good example of it can be positional encoding between (0,1).

**Learned Encoding**  
Another common way to add position information is with learned positional encodings. Here, each position in the sequence gets its own trainable vector, learned together with the rest of the model instead of being hand-designed. Because these position vectors are not shared across positions, permuting the tokens changes their positions and thus breaks permutation invariance, which is exactly what we want from a positional encoding. But the downside is that this scheme doesn’t naturally generalize to positions beyond those seen in training, meaning newer positions just have untrained embeddings. So, learned positional encoding are mainly a good fit when sequence lengths stay roughly the same during training and inference.

## References
- Bishop, C. M., & Bishop, H. (2023). Transformers. In Deep Learning: Foundations and Concepts (pp. 357-406). Cham: Springer International Publishing.
