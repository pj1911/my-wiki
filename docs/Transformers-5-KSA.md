## Introduction

The Kronecker product is an operation on two matrices that builds a larger block matrix, much like matrix multiplication combines matrices to produce a new one, but instead of taking row-by-column dot products, it multiplies every entry of the first matrix by the entire second matrix. If \(A=[a_{ij}] \in \mathbb{R}^{m\times n}\) and \(B \in \mathbb{R}^{p\times q}\), then their Kronecker product is written as \(A \otimes B\) and is defined by

$$
A \otimes B =
\begin{bmatrix}
a_{11}B & a_{12}B & \cdots & a_{1n}B \\
a_{21}B & a_{22}B & \cdots & a_{2n}B \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1}B & a_{m2}B & \cdots & a_{mn}B
\end{bmatrix},
$$

so each scalar entry of \(A\) is replaced by a copy of \(B\) scaled by that entry, so the result has shape \((mp)\times(nq)\).

More generally, if we take the Kronecker product of \(k\) square matrices \(A_1,\dots,A_k\) with sizes \(n_1\times n_1,\dots,n_k\times n_k\), then \(A_1\otimes\cdots\otimes A_k\) has size \(\bigl(\prod_{i=1}^k n_i\bigr)\times\bigl(\prod_{i=1}^k n_i\bigr)\). So for instance \(k\) copies of a \(2\times2\) matrix produce a \(2^k\times2^k\) output. This construction is widely used in linear algebra, tensor computations, signal processing, and quantum mechanics.

## Notations

We follow the following convention for the notation. This is similar to the notation used in chapter 1 of transformers on this website.

- The input is a sequence of \(N\) tokens, each token \(\mathbf{x}_n \in \mathbb{R}^{D}\), is stacked row-wise into

$$
\mathbf{X} \in \mathbb{R}^{N\times D}, \qquad \text{ where D represents the number of features}
$$

- Trainable projections single head form:

$$
\mathbf{Q}=\mathbf{X}\mathbf{W}^{(q)},\quad \mathbf{K}=\mathbf{X}\mathbf{W}^{(k)},\quad \mathbf{V}=\mathbf{X}\mathbf{W}^{(v)}.
$$

- Scaled dot-product attention row-wise softmax:

$$
\mathbf{Y}=\mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})
=\mathrm{Softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{D_k}}\right)\mathbf{V}, \qquad \mathbf{Y}\in\mathbb{R}^{N\times D}
$$

**Multi-head Attention.** With \(H\) heads, commonly \(D_k=D_v=D/H\). Each head \(h\) has its own \(\mathbf{W}^{(q)}_h,\mathbf{W}^{(k)}_h,\mathbf{W}^{(v)}_h\), producing head outputs \(\mathbf{H}_h \in \mathbb{R}^{N\times D_v}\), then

$$
\mathbf{Y}=\mathrm{Concat}[\mathbf{H}_1,\dots,\mathbf{H}_H]\mathbf{W}^{(o)},\qquad \mathbf{W}^{(o)}\in\mathbb{R}^{(H D_v)\times D},
\qquad \mathbf{Y}\in\mathbb{R}^{N\times D}
$$

## Why Kronecker-structured attention?

If the input is multiway (tensor-structured) rather than a 1D sequence, a naive Transformer approach is:

1. flatten the tensor into a sequence of length \(N\),
2. apply standard attention with an \(N\times N\) attention matrix,
3. reshape the output back.

This has the familiar quadratic attention bottleneck in both time and memory, and flattening also discards the explicit multi-axis structure. But the Kronecker-structured attention targets the case where token positions are naturally indexed by multiple axes (e.g., height \(\times\) width, or time \(\times\) variables) and tries to:

- preserve tensor structure,
- avoid forming a full \(N\times N\) attention matrix,
- still model cross-axis interactions.

**From flat sequence to multi-axis indexing.** In standard Transformer we take an input matrix

$$
\mathbf{X}\in\mathbb{R}^{N\times D},
$$

i.e. \(N\) tokens (rows), each with \(D\) features.
Kronecker-structured attention is useful when those \(N\) tokens actually come from a grid structure with \(k\) axes.
Concretely, assume there exist integers \(N_1,\dots,N_k\), then instead of a single positional index along a 1D sequence, the position is specified by \(k\) coordinates (one per axis).
Equivalently, we can reshape \(\mathbf{X}\) into a tensor

$$
\mathcal{X}=\text{reshape}(\mathbf{X})\in\mathbb{R}^{N_1\times\cdots\times N_k\times D}, \qquad
\text{ with } N = N_1 \times\cdots\times N_k
$$

where \(\mathcal{X}_{n_1,\dots,n_k,:}\in\mathbb{R}^D\) is the same token vector as some row of \(\mathbf{X}\), with structured indices.

**Example (2D image).** If we have a \(H\times W\) patch grid, for this case \(k=2\), \(N_1=H\), \(N_2=W\), and \(N=HW\).
Flattening gives \(\mathbf{X}\in\mathbb{R}^{(HW)\times D}\), while reshaping gives \(\mathcal{X}\in\mathbb{R}^{H\times W\times D}\).
In \(k\)-dimension, the positional index will be a multi-index \((n_1,\dots,n_k)\) with sizes \(N_1,\dots,N_k\).
This means, we store the tokens in a tensor

$$
\mathcal{X}\in \mathbb{R}^{N_1\times \cdots \times N_k \times D}.
$$

The last dimension is the feature dimension as usual. Note, we can reshape \(\mathcal{X}\) into the usual matrix \(\mathbf{X}\in\mathbb{R}^{N\times D}\) to recover standard attention.

## Kronecker-structured attention (KSA)

For baseline, we will compare against standard (flattened) multi-head self-attention applied to the data, i.e. treat the \(N=\prod_{i=1}^k N_i\) positions as an ordinary length-\(N\) sequence and compute attention in the usual way and compare it with the KSA mechanism.

### Multi-head attention on flattened multiway data (baseline)

For head \(h\) (width \(D_h\), typically \(D_h=D/H\)), we can compute

$$
\mathbf{Q}_h=\mathbf{X}\mathbf{W}^{(q)}_h,\quad
\mathbf{K}_h=\mathbf{X}\mathbf{W}^{(k)}_h,\quad
\mathbf{V}_h=\mathbf{X}\mathbf{W}^{(v)}_h,
$$

and

$$
\mathbf{H}_h=\mathrm{Softmax}\!\left(\frac{\mathbf{Q}_h\mathbf{K}_h^\top}{\sqrt{D_h}}\right)\mathbf{V}_h,
$$

which costs \(\mathcal{O}(N^2 D_h)\) per head and stores \(\mathcal{O}(N^2)\) attention coefficients. For Kronecker structured attention, the key idea is to represent (or approximate) the full attention matrix for a multiway input as a Kronecker composition of mode-wise attention matrices. The steps for doing this are given below:

#### Step 1: Q/K/V as tensors

Instead of viewing \(\mathbf{Q}_h,\mathbf{K}_h,\mathbf{V}_h\) as \(N\times D_h\) matrices (with \(N=\prod_{i=1}^k N_i\)), we reshape them into tensors

$$
\mathcal{Q}_h,\mathcal{K}_h,\mathcal{V}_h \in \mathbb{R}^{N_1\times \cdots \times N_k \times D_h},
$$

so that each multi-index \((n_1,\dots,n_k)\) corresponds to a single position and
\(
\mathcal{Q}_h[n_1,\dots,n_k,:]\in\mathbb{R}^{D_h}
\)
is the query vector at that position (and similarly for \(\mathcal{K}_h,\mathcal{V}_h\)). Concretely, for queries we have (showing shapes explicitly)

$$
\underbrace{\mathbf{q}_{h;n_1,\dots,n_k}}_{\in\mathbb{R}^{1\times D_h}}
\;=\;
\underbrace{\mathbf{x}_{n_1,\dots,n_k}}_{\in\mathbb{R}^{1\times D}}
\underbrace{\mathbf{W}^{(q)}_h}_{\in\mathbb{R}^{D\times D_h}},
$$

and the key/value equations are identical in form with \(\mathbf{W}^{(k)}_h\) and \(\mathbf{W}^{(v)}_h\).
Note, no new trainable weights are introduced here.

#### Step 2: Pool along all axes except one (to get axis-wise Q/K)

Recall \(\mathcal{Q}_h[n_1,\dots,n_k,:]\in\mathbb{R}^{D_h}\) is the query vector at position \((n_1,\dots,n_k)\). For each axis (mode) \(i\in\{1,\dots,k\}\), we want a compressed set of queries/keys that depends only on the coordinate along that axis.
We therefore build

$$
\tilde{\mathbf{Q}}^{(i)}_h \in \mathbb{R}^{N_i\times D_h},\qquad
\tilde{\mathbf{K}}^{(i)}_h \in \mathbb{R}^{N_i\times D_h,}
$$

where, both contain one \(D_h\)-dimensional vector for each possible value of the \(i\)-th coordinate (here, \(i=1\) could mean height, \(i=2\) width and so on). A simple permutation-invariant pooling (used commonly in Higher Order Transformers (HOT)) is: fix \(n_i\) and sum over all other coordinates:

$$
\tilde{\mathbf{Q}}^{(i)}_h[n_i,:] =
\sum_{n_1=1}^{N_1}\cdots
\sum_{n_{i-1}=1}^{N_{i-1}}
\sum_{n_{i+1}=1}^{N_{i+1}}
\cdots
\sum_{n_k=1}^{N_k}
\mathcal{Q}_h[n_1,\dots,n_{i-1},\,n_i,\,n_{i+1},\dots,n_k,:],
\qquad n_i\in\{1,\dots,N_i\}.
$$

The computation looks similar for \(\tilde{\mathbf{K}}^{(i)}_h[n_i,:]\), where \(n_i\) is the coordinate value along that axis (e.g. a particular height or width index).

**Example (3D grid).** Let \(\mathcal{Q}_h\in\mathbb{R}^{H\times W\times T\times D_h}\). Pooling along all axes except one produces
one \(D_h\)-vector per coordinate value on that axis (i.e. a unique row for each \(h\), each \(w\), and each \(t\)):

$$
\tilde{\mathbf{Q}}^{(1)}_h[h,:]=\sum_{w=1}^W\sum_{t=1}^T \mathcal{Q}_h[h,w,t,:],
\quad h\in\{1,\dots,H\}
\;\;\Rightarrow\;\;
\tilde{\mathbf{Q}}^{(1)}_h\in\mathbb{R}^{H\times D_h},
$$

$$
\tilde{\mathbf{Q}}^{(2)}_h[w,:]=\sum_{h=1}^H\sum_{t=1}^T \mathcal{Q}_h[h,w,t,:],
\quad w\in\{1,\dots,W\}
\;\;\Rightarrow\;\;
\tilde{\mathbf{Q}}^{(2)}_h\in\mathbb{R}^{W\times D_h},
$$

$$
\tilde{\mathbf{Q}}^{(3)}_h[t,:]=\sum_{h=1}^H\sum_{w=1}^W \mathcal{Q}_h[h,w,t,:],
\quad t\in\{1,\dots,T\}
\;\;\Rightarrow\;\;
\tilde{\mathbf{Q}}^{(3)}_h\in\mathbb{R}^{T\times D_h}.
$$

(And the same construction applies to \(\tilde{\mathbf{K}}^{(i)}_h\)).

#### Step 3: Mode-wise attention matrices

Now compute a standard scaled dot-product attention matrix per axis (mode):

$$
\mathbf{A}^{(i)}_h =
\mathrm{Softmax}\!\left(
\frac{\tilde{\mathbf{Q}}^{(i)}_h(\tilde{\mathbf{K}}^{(i)}_h)^\top}{\sqrt{D_h}}
\right)
\in \mathbb{R}^{N_i\times N_i}.
$$

Each \(\mathbf{A}^{(i)}_h\) is row-stochastic because the softmax is applied row-wise.

**Example (3D case, axis \(W\)).** For \(\mathcal{Q}_h,\mathcal{K}_h\in\mathbb{R}^{H\times W\times T\times D_h}\), the pooled matrices along the \(W\)-axis are

$$
\tilde{\mathbf{Q}}^{(2)}_h\in\mathbb{R}^{W\times D_h},\qquad \tilde{\mathbf{K}}^{(2)}_h\in\mathbb{R}^{W\times D_h}.
$$

The mode-wise attention over width positions is then

$$
\mathbf{A}^{(2)}_h =
\mathrm{Softmax}\!\left(\frac{\tilde{\mathbf{Q}}^{(2)}_h(\tilde{\mathbf{K}}^{(2)}_h)^\top}{\sqrt{D_h}}\right)
\in\mathbb{R}^{W\times W},
$$

so each row \(\mathbf{A}^{(2)}_h[w,:]\) is a distribution over all width indices \(w'\in\{1,\dots,W\}\).

#### Step 4a: Kronecker product attention (HOT-product)

Define the full attention matrix for head \(h\) as

$$
\mathbf{A}_h
\;\approx\;
\mathbf{A}^{(1)}_h \otimes \mathbf{A}^{(2)}_h \otimes \cdots \otimes \mathbf{A}^{(k)}_h
\quad\in\mathbb{R}^{N\times N}.
$$

If each factor is row-stochastic, their Kronecker product is also row-stochastic (so it is a valid attention weight matrix).

#### Step 4b: Kronecker sum attention (HOT-sum)

Using the Kronecker sum idea, we can also build a row-stochastic combination like

$$
\mathbf{A}_h
\;\approx\;
\frac{1}{k}\sum_{i=1}^k
\bigl(\mathbf{I}_{N_1}\otimes \cdots \otimes \mathbf{I}_{N_{i-1}}\otimes \mathbf{A}^{(i)}_h
\otimes \mathbf{I}_{N_{i+1}}\otimes \cdots \otimes \mathbf{I}_{N_k}\bigr).
$$

The \(1/k\) normalization keeps row sums equal to \(1\).

#### Step 5: Applying Kronecker attention without constructing the full \(N\times N\) matrix

In the Kronecker product variant, the full attention for head \(h\) is conceptually

$$
\mathbf{A}_h \;\approx\; \mathbf{A}^{(1)}_h\otimes\cdots\otimes \mathbf{A}^{(k)}_h \in \mathbb{R}^{N\times N},
\qquad N=\prod_{i=1}^k N_i.
$$

However, explicitly forming \(\mathbf{A}_h\) is infeasible: it would require storing \(N^2\) entries, which is exactly the quadratic bottleneck we are trying to avoid.
The key trick is that we never need \(\mathbf{A}_h\) itself, we only need its action on values.

Let \(\mathcal{V}_h\in\mathbb{R}^{N_1\times\cdots\times N_k\times D_h}\) be the value tensor.
The Kronecker structure implies the following identity:

$$
\bigl(\mathbf{A}^{(1)}_h\otimes\cdots\otimes \mathbf{A}^{(k)}_h\bigr)\,\mathrm{vec}(\mathcal{V}_h) =
\mathrm{vec}\!\Bigl(\mathcal{V}_h \times_1 \mathbf{A}^{(1)}_h \times_2 \mathbf{A}^{(2)}_h \cdots \times_k \mathbf{A}^{(k)}_h\Bigr),
$$

where \(\times_i\) denotes multiplication along the \(i\)-th axis (mode) of the tensor.
Therefore we compute the head output by sequentially applying the small mode-wise attention matrices:

$$
\mathcal{H}_h =
\bigl(\cdots((\mathcal{V}_h \times_1 \mathbf{A}^{(1)}_h)\times_2 \mathbf{A}^{(2)}_h)\cdots \times_k \mathbf{A}^{(k)}_h\bigr).
$$

This uses only the factors \(\mathbf{A}^{(i)}_h\in\mathbb{R}^{N_i\times N_i}\) (memory \(\sum_i N_i^2\)) and avoids storing \(\mathbf{A}_h\) (memory \(N^2\)).

**Sum variant.** If instead we use a Kronecker-sum style mixture, the action on \(\mathcal{V}_h\) is a simple average of \(k\) mode-wise applications:

$$
\mathcal{H}_h
\approx
\frac{1}{k}\sum_{i=1}^k \bigl(\mathcal{V}_h \times_i \mathbf{A}^{(i)}_h\bigr),
$$

again requiring only the small \(\mathbf{A}^{(i)}_h\) matrices and never constructing any \(N\times N\) attention map.

**Why this is better (what we save).** Full attention would require materializing an attention map with \(N^2\) entries per head.
Kronecker-structured attention stores only \(\sum_{i=1}^k N_i^2\) entries per head and applies them through tensor-matrix multiplies, so the attention-map memory drops from quadratic in \(N\) to quadratic in the per-axis sizes.

### Output projection (same as multi-head attention)

After computing each head tensor \(\mathcal{H}_h\in\mathbb{R}^{N_1\times\cdots\times N_k\times D_h}\),
apply the usual output projection along the last dimension:

$$
\mathcal{Y} =
\sum_{h=1}^H \mathcal{H}_h \times_{k+1}\mathbf{W}^{(o)}_h,
\qquad \mathbf{W}^{(o)}_h\in\mathbb{R}^{D_h\times D}.
$$

Again: no new trainable parameters beyond \(\mathbf{W}^{(q)}_h,\mathbf{W}^{(k)}_h,\mathbf{W}^{(v)}_h,\mathbf{W}^{(o)}_h\).

## Complexity comparisons

### Trainable parameters (Q/K/V/O)

Counting only the attention block projections (ignoring biases), the number of parameters are given by:

$$
\sum_{h=1}^H
\left(
\underbrace{D D_h}_{\mathbf{W}^{(q)}_h} +
\underbrace{D D_h}_{\mathbf{W}^{(k)}_h} +
\underbrace{D D_h}_{\mathbf{W}^{(v)}_h} +
\underbrace{D_h D}_{\mathbf{W}^{(o)}_h}
\right) =
4 H D D_h.
$$

With the common choice \(D_h=D/H\):

params \(= 4D^2\).

Kronecker-structured attention does not reduce these trainable parameters;
it reduces the cost of forming/applying the data-dependent attention weights.

### Attention coefficients stored/computed (not trainable parameters)

This is often what people mean informally when discussing efficiency:

- Full attention stores an \(N\times N\) matrix per head: \(N^2\) coefficients, with \(N=\prod_i N_i\).
- Kronecker attention stores \(k\) smaller matrices: \(\sum_{i=1}^k N_i^2\) coefficients per head.

**Example (2D).** If \(N_1=N_2=64\), then \(N=4096\):

$$
N^2 = 16{,}777{,}216
\qquad\text{vs}\qquad
N_1^2+N_2^2 = 8192.
$$

That is a massive reduction in attention-map memory.

## Inductive bias and limitations of Kronecker-structured attention

Kronecker-structured attention imposes an explicit separability assumption across axes. In the product form,

$$
\mathbf{A}_h \approx \mathbf{A}^{(1)}_h\otimes\cdots\otimes \mathbf{A}^{(k)}_h,
$$

the attention weight between positions \((n_1,\dots,n_k)\) and \((m_1,\dots,m_k)\) factorizes as

$$
\mathbf{A}_h\bigl[(n_1,\dots,n_k),(m_1,\dots,m_k)\bigr]
\approx
\prod_{i=1}^k \mathbf{A}^{(i)}_h[n_i,m_i].
$$

Thus, affinity along one axis is combined multiplicatively with affinity along the others: a weak match on any single axis can substantially reduce the total weight.

**Inductive bias.**

- Axis-wise separability.
  The model is biased toward representing interactions through per-axis affinities, rather than arbitrary joint couplings across axes.

- Preference for structured, tensor-aligned relations.
  This is well suited to data whose dependencies decompose naturally across modes (for example, time and space), but less suited to relations that are inherently cross-axis, such as diagonal, rotated, or other joint geometric patterns.

- Restricted per-head expressivity.
  Each head contributes a separable attention pattern, so the overall multi-head mechanism can only combine a finite number of such components. This controls capacity and can improve efficiency, but it also limits the class of attention maps that can be represented compactly.

**Limitations in practice.**

- Not every attention map is representable.
  A general \(N\times N\) attention matrix has far more degrees of freedom than a Kronecker-factored form, which uses only the parameters of the smaller mode-wise matrices. As a result, some attention patterns cannot be expressed exactly.

- Joint cross-axis interactions can be poorly modeled.
  When the true dependency depends on coupled coordinates, for example, \((h,w)\) attending to \((h+\Delta,w+\Delta)\), a separable factorization may require many heads or still yield a poor approximation.

- Product form can be brittle.
  When the mode-wise matrices \(\mathbf{A}^{(i)}_h\) are attention weights, their entries are nonnegative and typically at most \(1\), so

$$
\prod_{i=1}^k \mathbf{A}^{(i)}_h[n_i,m_i]\le \min_i \mathbf{A}^{(i)}_h[n_i,m_i].
$$

Hence a single small mode-wise weight can strongly suppress the full interaction, even if the other axes match well.

- Sum form is less selective.
  With

$$
s_{\mathrm{prod}}=\prod_{i=1}^k \mathbf{A}^{(i)}_h[n_i,m_i],
\qquad
s_{\mathrm{sum}}=\sum_{i=1}^k \mathbf{A}^{(i)}_h[n_i,m_i],
$$

the product is large only if all mode-wise weights are large, whereas the sum can be large whenever a few axes match strongly. Hence the sum is less sensitive to mismatches on individual axes, but it also enforces joint agreement less strongly, meaning a small and large combination can still remain large.

- Mode-wise pooling may lose context.
  The aggregated representations \(\tilde{\mathbf{Q}}^{(i)}_h\) and \(\tilde{\mathbf{K}}^{(i)}_h\) summarize over the other axes, which can remove localized or configuration-specific information.

- Choice of axes is critical.
  The method works best when the chosen factorization matches the true structure of the data. An inappropriate choice of \(k\) or of how dimensions are grouped can reduce performance.

## References

1. P. Chauhan, Transformers -- Introduction (Attention, Scaled Attention, Multi-Head Attention).  
   https://pj1911.github.io/my-wiki/Transformers-1-Introduction/

2. S. Omranpour, R. Rabbany, G. Rabusseau, Higher-Order Transformers (HOT) with Kronecker Factorized Attention (arXiv), arXiv preprint arXiv:2412.02919.


