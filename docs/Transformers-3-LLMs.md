# Transformer Language Models

## Introduction

The transformer layer is a very flexible building block for neural networks, and it works especially well for natural language. When we scale transformers up, we get massive neural networks called *large language models (LLMs)*, which have turned out to be remarkably capable.

We can use transformers for many different language tasks, and it helps to think of them in three main categories, based on the kinds of inputs and outputs they handle:

- **Encoder models.** Here the input is a sequence of words, and the output is a single value or label. For example, in sentiment analysis we feed in a sentence and output one variable that describes its sentiment, such as *happy* or *sad*. In this setup, the transformer acts as an *encoder* of the input sequence.

- **Decoder models.** In this case the input is a single vector, and the output is a sequence of words. A common example is image captioning, where an input image is first mapped to a vector, and then a transformer *decoder* takes that vector and generates a text caption word by word.

- **Encoder–decoder (sequence-to-sequence) models.** Here both the input and the output are word sequences. A typical example is machine translation, where the model takes a sentence in one language as input and produces a sentence in another language as output. In this setup, we use transformers in both roles: an encoder for the input sequence and a decoder for the output sequence.

In the rest of this chapter, we will look at each of these three classes of language model in turn, using example architectures to show how they are built.

## Decoder transformers

We now look at decoder-only transformer models. These are used as
generative models: given a prefix of a sequence, they generate the rest,
token by token. A key example is the GPT family (generative pre-trained transformer). GPT uses the
transformer architecture to build an autoregressive model, where each conditional distribution

$$
p(x_n \mid x_1,\ldots,x_{n-1})
$$

is represented by the same transformer neural network (meaning same parameters) learned from data. At step \(n\), the model conceptually takes the first \(n-1\) tokens as input and
outputs the conditional distribution over the vocabulary of size \(K\) for token \(n\). Sampling from this
distribution extends the sequence to length \(n\), and we can repeat this process
to get token \(n+1\), \(n+2\), and so on, up to a maximum sequence length set by
the transformer. This token-by-token view is how we *generate* text. Training follows a different paradigm defined later in this chapter.

### Network architecture and self-supervised training.

A GPT-style model is a stack of transformer layers. The input is a sequence of
token embeddings

$$
x_1,\ldots,x_N,
$$

with each \(x_n \in \mathbb{R}^D\). Stacking these row-wise gives
\(X \in \mathbb{R}^{N\times D}\). The transformer stack maps \(X\) to a sequence of
hidden vectors

$$
\tilde{x}_1,\ldots,\tilde{x}_N,
$$

collected in \(\tilde{X} \in \mathbb{R}^{N\times D}\). Each \(\tilde{x}_n\) is the
hidden representation used to predict \(x_n\) given the prefix \(x_{1:n-1}\). At every position we want a probability distribution over a vocabulary of \(K\)
tokens, but the transformer outputs \(D\)-dimensional vectors. We therefore apply
the *same* linear layer at all positions, with weight matrix
\(W^{(p)} \in \mathbb{R}^{D\times K}\), followed by a softmax:

$$
Y = Softmax\!\bigl(\tilde{X} W^{(p)}\bigr) \in \mathbb{R}^{N\times K},
$$

so that the \(n\)th row \(y_n^\top\) is a full probability distribution over the
\(K\) vocabulary items and is the model’s prediction for token
\(x_n\) given \(x_{1:n-1}\).

**Self supervised training.**

We train this model on a large corpus of raw text using a
self-supervised objective, where the input text itself provides the output targets.
To turn text into something the model can handle, we first map each token to an integer in a fixed vocabulary.
Let \(t_n \in \{1,\ldots,K\}\) be the index of \(x_n\) in the vocabulary.
Then for a single sequence, the loss is the sum of cross-entropies over all
positions and can be written as:

$$
\mathcal{L}
  = - \sum_{n=1}^{N} \log y_{n,t_n}.
$$

This means:

- At each position \(n\), the model outputs a vector
  
$$
y_n \in \mathbb{R}^K,
$$

  which is the \(n\)th row of \(Y\). We can write it as
  
$$
y_n = (y_{n,1}, y_{n,2}, \ldots, y_{n,K}),
$$
  
  where \(y_{n,k}\) is the predicted probability that the \(n\)th token is the
  \(k\)th vocabulary item.

- The true token at position \(n\) is \(x_n\). We represent it by its
  vocabulary index \(t_n \in \{1,\ldots,K\}\) (for example, if “river” is the
  57th word in the vocabulary, then \(t_n = 57\) for that position). Then
  
$$
y_{n,t_n} = \text{``model’s predicted probability of the true token at step } n\text{''}.
$$

- The cross-entropy loss at that position \(n\) is then given by:
  
$$
-\log y_{n,t_n},
$$
  
  which is large if the model puts low probability on the correct word, and
  small if it puts high probability on it, and we want to minimize this loss.

Summing over all positions \(n=1,\ldots,N\) gives the total loss
\(\mathcal{L}\) for that one sequence where every token in the sequence contributes
one term: \(-\log y_{n,t_n}\).

Over the whole dataset we have many sequences (sentences, or longer chunks).
We usually treat them as i.i.d. samples and sum the same loss over all of
them. If we index the many sequences by \(s=1,\ldots,S\), with length \(N^{(s)}\), then
the total loss is

$$
\mathcal{L}_{\text{total}}
  = \sum_{s=1}^{S} \sum_{n=1}^{N^{(s)}} - \log y^{(s)}_{n,t^{(s)}_n},
$$

where \(y^{(s)}_{n,\cdot}\) is the prediction vector for position \(n\) in
sequence \(s\), and \(t^{(s)}_n\) is the true vocabulary index at that position. Intuitively, each token plays two roles:

- it is a *target* we want the model to predict correctly, and
- it is part of the *prefix* that helps predict later tokens.

For example, for the sentence

> I swam across the river to get to the other bank.

we embed all tokens into a matrix \(X\) and feed it through the transformer to get
predictions \(y_1,\ldots,y_N\). At the position of the word “the”, the input
effectively corresponds to the prefix “I swam across”, and the model output
\(y_n\) should put high probability on the vocabulary index of “the”. At the
next position, the input prefix is “I swam across the”, and the output
\(y_{n+1}\) should place high probability on “river”, and so on for all later
positions.

However, if during training we let the network use the *entire* sentence
at every position, then when predicting the \(n\)th token it can also see token
\(n\) itself (and even later tokens). In that case it can simply learn to copy
the next word from the input instead of genuinely modelling
\(p(x_n \mid x_{1:n-1})\). This behaviour would give a very low training loss,
but it would fail at generation time, where future tokens are not available.
In the next section we will see how the architecture is constrained so that
each prediction can only depend on earlier tokens in the sequence.

### Preventing cheating: shifting and masking.

We prevent the above issue in two ways:

1. **Shifted inputs.** We shift the input sequence one step to the
   right. Input token \(x_n\) now corresponds to output \(y_{n+1}\), with target
   \(x_{n+1}\). We prepend a special start-of-sequence token, \(\langle\text{start}\rangle\), at the first input position. Even with this shift, a single training sequence \((x_1,\dots,x_T)\) still
   yields many input–target pairs: for each \(n \ge 1\) we treat the prefix
   \((x_1,\dots,x_{n-1})\) (or, after shifting, \((\langle\text{start}\rangle,x_1,\dots,x_{n-1})\)) as the input and \(x_n\) as the corresponding target.

   **Before shifting.** Conceptually, we were thinking of many separate
   next-token training pairs, e.g.

 $$
 (\text{``I''} \to \text{``swam''}),\quad
 (\text{``I swam''} \to \text{``across''}),\quad \ldots
 $$

   Each pair would be run through the model as its own little training example.

   **What we change.** Instead of \(N\) separate runs for one sentence, we
   pack everything into a single sequence:

   - Start from the raw text tokens

 $$
 x_1, x_2, \ldots, x_N.
 $$

   - Build the *input row* by shifting right and inserting the start
     token:

 $$
 \underbrace{\langle\text{start}\rangle, x_1, x_2, \ldots, x_{N-1}}_{\text{inputs}}
 $$

   - Build the *target row* by shifting left:

 $$
 \underbrace{x_1, x_2, \ldots, x_N}_{\text{targets}}.
 $$

   So compared to the original text, we have:

   - removed \(x_N\) from the input side,
   - added \(\langle\text{start}\rangle\) at the front,
   - kept the targets as the original sequence.

   With this layout, column \(n\) of the model sees a prefix ending at \(x_{n-1}\)
   and is trained to predict \(x_n\). All \(N\) next-token prediction tasks are now
   done in one forward pass, and the masking step (described next) makes sure
   each position only uses past tokens.

2. **Masked (causal) attention, padding, and efficient generation.**

   **Before masking.** In plain self-attention, every token can attend to
   every other token in the sequence. If we used this directly for
   next-token prediction, token \(n\) could see token \(n+1\) and simply copy it,
   which is useless at generation time when \(x_{n+1}\) is not known.

   **What we change (causal mask).** We force each token to look only at
   itself and earlier tokens:

   - In the attention matrix \(\text{Attention}(Q,K,V)\) we zero out all entries
     where a token would attend to any *later* position.
     
   - In practice we set the corresponding logits to \(-\infty\), so the
     attention softmax gives probability almost \(0\) there and renormalizes over the allowed
     positions.

   The result is a lower-triangular attention matrix: row \(n\) only uses columns
   \(1,\dots,n\).

   **Handling different lengths.** Real sentences have different lengths, but GPUs work
   best if we process many sequences together as one batch tensor. Without care,
   shorter sequences would leave random or empty slots that other tokens might
   attend to.

   **What we change (padding mask).**

   - We pad shorter sequences with a special token
     \(\langle\text{pad}\rangle\) so all sequences share the same length.
     
   - We add a second mask that blocks attention to any position containing
     \(\langle\text{pad}\rangle\). This mask is specific to each sequence in the
     batch.

   **Caching.** During generation, we repeatedly:

   1. feed the current prefix into the model,
   2. use the softmax output to get a distribution over the next token,
   3. sample or choose a token, append it, and repeat.

   Naively, this means re-running the whole transformer on the entire prefix for
   every new token.

   **What we change.** Because of the causal mask,
   the representation of token \(i\) depends only on tokens \(1,\dots,i\) and never
   on future tokens. So when we extend the sequence:

   - At step \(t\) we run the full model once for \(x_1,\dots,x_t\) and cache
     per-layer key and value tensors \(K^{(\ell)}_1,\dots,K^{(\ell)}_t\) and
     \(V^{(\ell)}_1,\dots,V^{(\ell)}_t\). Note that we do not cache queries because each query is only used once for its own token at that time step and is never reused later, so caching it would waste memory without reducing computation.

   - At step \(t+1\) we keep these cached states fixed (earlier tokens are
     not allowed to change). We only compute the new token’s hidden states and
     its \(Q^{(\ell)}_{t+1},K^{(\ell)}_{t+1},V^{(\ell)}_{t+1}\), then run
     attention for position \(t+1\) using the cached keys/values plus the new ones.

   Conceptually we still “run the model” at each step, but most computation is
   reused, making long-sequence generation practical.

### How shifting and masking work together

Shifted inputs and causal masking are two separate ideas, but in practice
*they are always used together* in decoder-only language models. The
typical order each forward pass is:

$$
\text{raw tokens} \;\Rightarrow\; \text{shifted inputs + targets}
\;\Rightarrow\; \text{build mask from positions}
\;\Rightarrow\; \text{run transformer}.
$$

Below is the same sentence at each stage.

**Step 0: The naive case.**

Raw token sequence:

$$
(x_1, x_2, \dots, x_N).
$$

If we fed this directly into full self-attention, each position \(n\) could attend
to \(x_{n+1}, \dots, x_N\) and just copy the next token. This would lead to data leakage.

**Step 1: Apply shifted inputs.**

First we turn the raw sequence into an input row and a target row:

$$
\begin{aligned}
\text{inputs} &: (\langle\text{start}\rangle, x_1, x_2, \dots, x_{N-1}), \\
\text{targets} &: (x_1, x_2, \dots, x_N).
\end{aligned}
$$

Now a single forward pass gives us \(N\) next-token predictions in parallel:
at position \(n\) the model outputs a distribution meant to match target \(x_n\). At this stage, if attention were still fully unmasked, position \(n\) could
*still* peek at later inputs and cheat.

**Step 2: Add causal masking.**

We keep the shifted inputs and targets exactly as above. What we change now is
*only* the attention pattern. We build a mask matrix \(M \in \{0,-\infty\}^{N \times N}\):

$$
M_{ij} =
\begin{cases}
0      & \text{if } j \le i, \\
-\infty & \text{if } j > i.
\end{cases}
$$

This mask is added to the attention logits before the softmax. The final layout looks like this:

$$
\begin{array}{c|cccc}
\text{position} & 1 & 2 & \cdots & N \\
\hline
\text{input token}  & \langle\text{start}\rangle & x_1 & \cdots & x_{N-1} \\
\text{target token} & x_1                        & x_2 & \cdots & x_N     \\
\text{may attend to input position}& 1                          & 1,2 & \cdots & 1,\dots,N
\end{array}
$$

So in the final model:

- shifting decides *which token we try to predict at each position*,
- masking decides *which past tokens each position is allowed to use*.

Both are applied every time we run the transformer model.

