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

### Network architecture and self-supervised training

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

### Preventing cheating: shifting and masking

We prevent the above issue in two ways:

1. **Shifted inputs.** We shift the input sequence one step to the
   right. Input token \(x_n\) now corresponds to output \(y_{n+1}\), with target
   \(x_{n+1}\). We prepend a special start-of-sequence token, \(\langle\text{start}\rangle\), at the first input position. Even with this shift, a single training sequence \((x_1,\dots,x_T)\) still
   yields many input–target pairs: for each \(n \ge 1\) we treat the prefix
   \((x_1,\dots,x_{n-1})\) (or, after shifting, \((\langle\text{start}\rangle,x_1,\dots,x_{n-1})\)) as the input and \(x_n\) as the corresponding target.

   **Before shifting.** Conceptually, we were thinking of many separate
   next-token training pairs, e.g.

$$
(\text{I} \to \text{swam}),\quad
(\text{I swam} \to \text{across}),\quad \ldots
$$

   Each pair would be run through the model as its own little training example.

   **What we change.** Instead of \(N\) separate runs for one sentence, we
   pack everything into a single sequence:

- Start from the raw text tokens:

$$
x_1, x_2, \ldots, x_N.
$$

- Build the *input row* by shifting right and inserting the start token:

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

## Sampling strategies

As we saw a decoder transformer outputs, at each step, a probability distribution over the
next token. To extend a sequence we must turn this distribution into a concrete
choice. For this, several strategies are used.

### Greedy search

The simplest method, *greedy search*, always chooses the token with the
highest probability. This makes generation deterministic: the same input prefix
always produces the same continuation.

Note that choosing the most probable token at each step is *not* the same
as choosing the most probable overall sequence. The probability of a full
sequence \(y_1,\ldots,y_N\) is

$$
p(y_1,\ldots,y_N)=\prod_{n=1}^N p(y_n \mid y_1,\ldots,y_{n-1})
$$

If there are \(N\) steps and a vocabulary of size \(K\), the number of possible
sequences is \(\mathcal{O}(K^N)\), which grows exponentially with \(N\), so
exhaustively finding the single most probable sequence is infeasible. Greedy
search, by contrast, has cost \(\mathcal{O}(KN)\): at each of the \(N\) steps it
scores all \(K\) tokens once and picks the best, so the total work scales
linearly with \(N\).

### Beam search

To get higher-probability sequences than greedy search, we can use
*beam search*. Instead of keeping only one hypothesis, we maintain \(B\)
partial sequences at step \(n\) where \(B\) is the *beam width*. We feed all \(B\)
sequences through the model and, for each, consider the \(B\) most probable next
tokens. Since each of the \(B\) partial sequences can be extended in \(B\) ways, this yields \(B \cdot B = B^2\) candidate sequences, from which we keep the \(B\)
sequences with the highest total sequence probability. The algorithm therefore
tracks \(B\) alternatives and their probabilities at all times, and finally
returns the most probable sequence among them. For example, with \(B=2\) and
current beam \{ `I`, `You` \}, if the top two continuations for
each are \{ `am`,`like` \} and \{ `are`,`like` \},
the \(B^2=4\) candidates are \{ `I am`,`I like`,`You are`,`You like` \}, from which we keep the best \(B=2\).

Because the probability of a sequence is a product of stepwise probabilities,
and each probability is at most one, long sequences tend to have lower raw
probability than short ones. Beam search is therefore usually combined with a
length normalization so that different sequence lengths can be compared fairly.
Its computational cost is \(\mathcal{O}(BKN)\), still linear in \(N\) but \(B\) times
more expensive than greedy search. For very large language models this extra
factor can make beam search unattractive.

### Diversity and randomness

Greedy and beam search both focus on high-probability sequences, but this often
reduces diversity and can even cause loops in which the same subsequence is
repeated. Human-written text is often more surprising (lower probability under
the model) than automatically generated text.

An alternative is to sample the next token directly from the softmax
distribution at each step: instead of always taking the single most probable
token (the argmax), we treat the softmax output as a categorical distribution
over the \(K\) tokens and randomly draw one token according to these
probabilities. This is same as picking one of the \(K\) outputs from
the final softmax at each step, but doing so stochastically according to their
probabilities rather than deterministically choosing the largest. This can give diverse outputs, but with a large
vocabulary the distribution typically has a long tail of very low-probability
tokens, and sampling from the full distribution can easily pick poor choices.

### Top-\(K\) and nucleus sampling

To balance between determinism and randomness, we can restrict sampling to the
most likely tokens. In *top-\(K\) sampling* we keep only the \(K\) tokens with
highest probability, renormalize their probabilities, and sample from this
reduced set. This removes the very low-probability tail, which cuts down on
wild or nonsensical tokens but still allows multiple plausible choices.

A popular variant is *top-\(p\)* or *nucleus sampling*. Here we choose
the smallest set of tokens whose cumulative probability reaches a threshold \(p\),
then renormalize and sample only from that subset. Unlike top-\(K\), the size of
this set adapts to the model’s confidence: when the model is sure, the nucleus
is small and more focused; when it is uncertain, the nucleus becomes larger and
more diverse.

### Temperature

A softer way to control randomness is to introduce a temperature parameter \(T\)
into the softmax:

$$
y_i = \frac{\exp(a_i/T)}{\sum_j \exp(a_j/T)}
$$

We then sample the next token from this modified distribution.

- \(T=0\) concentrates all probability on the most likely token
  (greedy selection).
- \(T=1\) recovers the original softmax distribution.
- \(T \to \infty\) gives a uniform distribution over all tokens.
- For \(0 < T < 1\), probability mass is pushed towards higher-probability
  tokens.

**Training vs. generation (exposure bias).**

During training, the model sees *human*-generated sequences as input.
During generation, however, the input prefix is itself model-generated. Over
time, this mismatch can cause the model to drift away from the distribution of
sequences present in the training data, which is an important challenge in
sequence generation.

## Encoder transformers

Encoder-based transformer language models take a whole sequence as input and
turn it into one or more fixed-size vectors. These vectors can then be used to
predict a discrete category (a *class label*), such as *positive* vs.\
*negative* sentiment, or *spam* vs. *not spam*. In other
words, they *encode* the entire sentence into one or more summary
representations, but they do not generate text by themselves. This contrasts
with *decoder* models, which are trained to predict the next token and can
therefore generate sequences, and with *encoder–decoder* models, which
first encode an input sequence and then use a decoder to generate a separate
output sequence (as in machine translation). A key example
is *BERT* (bidirectional encoder representations from transformers). The idea is:

- pre-train a transformer encoder on a huge text corpus,
- then apply *transfer learning* by fine-tuning it on many downstream
  tasks, each with a much smaller task-specific data set.

### Pre-training with masked tokens

Our goal in the pre-training stage is to teach the encoder a rich, general
understanding of language using only raw text (no human labels). To do this we
use a self-supervised prediction task that encourages the model to use both the
preceding and following words around a token, so it learns *bidirectional*
representations. This turns plain text into its own source of supervision:
by hiding some words and asking the model to guess them, we create huge numbers
of training examples for free while directly teaching it to understand how
words fit together in context.

BERT achieves this with a *masked language modelling* objective. Every
input sequence begins with a special token \(\langle\text{class}\rangle\) whose
output is ignored during pre-training but will be used later. The model is then
trained on sequences of tokens where a random subset (e.g. \(15\%\) of tokens) is
replaced by a special \(\langle\text{mask}\rangle\) token. The task is then to predict
the original tokens at the corresponding output positions. For example take the input sequence:

$$
\text{I } \langle\text{mask}\rangle \text{ across the river to get to the }
\langle\text{mask}\rangle \text{ bank.}
$$

The network should output “swam” at position \(2\) and “other” at position
\(10\) while all other outputs are ignored for computing the loss. As a result of bidirection:

- we do not shift inputs to the right (no autoregressive structure),
- we do not need causal masks to hide future tokens.

Compared with decoder models, this is less efficient for training, because only
a subset of tokens provide supervised targets, and the encoder alone cannot
generate sequences. If we always replaced the chosen tokens by \(\langle\text{mask}\rangle\) during pre-training, the model would mainly learn to handle inputs that contain many \(\langle\text{mask}\rangle\) symbols. At fine-tuning and test time, however, it is given normal sentences with no \(\langle\text{mask}\rangle\) tokens, so the input distribution looks very different from what it saw during pre-training. This *mismatch* can make the learned representations less useful, because the model has had much less practice dealing with real words in those positions. To reduce this gap, we can adjust the \(15\%\) selected tokens as
follows:

- \(80\%\) are replaced by \(\langle\text{mask}\rangle\),
- \(10\%\) are replaced by a random vocabulary token,
- \(10\%\) are left unchanged (but the model is still trained to predict
  them).

### Fine-tuning for downstream tasks

Once the encoder is pre-trained, we attach a task-specific output layer and
fine-tune the whole model.

- **Sequence-level classification (e.g. sentiment).**  
  The input can be a whole sentence or multiple paragraphs, tokenized and fed through
  the encoder with the \(\langle\text{class}\rangle\) token at the first
  position. After the final encoder layer we get one output vector for each
  input token \(h_0, h_1, \cdots , h_n\). The first one, \(h_{0} \in \mathbb{R}^D\) is treated as a summary of the entire sequence.

  To turn this summary into a label, we attach a small task-specific
  classifier on top. The simplest choice is a linear layer with parameter
  matrix \(W \in \mathbb{R}^{K \times D}\) and bias \(b \in \mathbb{R}^K\),
  giving logits

$$
z = W h_{\text{class}} + b .
$$

  For \(K\)-way classification we apply a softmax to \(z\) to get class
  probabilities. For binary classification (\(K=2\)) like positive or negative sentiment, a common variant is
  to use a single output score \(s = w^\top h_{\text{class}} + b\) followed
  by a logistic sigmoid. This linear head is just a minimal example, in practice we can replace
  it with a small MLP or any other differentiable module that maps
  \(h_{0}\) to the desired label space.

- **Token-level classification (e.g. tagging each word as person,
  place, colour, etc.).**  
  Here the input is again a full sequence as above with the \(\langle\text{class}\rangle\) token at the beginning. After passing this
  sequence through the encoder, we obtain one hidden vector for each input
  position as well:

$$
h_0, h_1, \ldots, h_N \in \mathbb{R}^D,
$$

  where \(h_0\) corresponds to \(\langle\text{class}\rangle\) and
  \(h_1,\ldots,h_N\) correspond to the actual tokens in the sentence.

  For token-level labelling we ignore \(h_0\) and attach the *same*
  linear classifier to each of the remaining hidden states. Concretely, we
  use a weight matrix \(W \in \mathbb{R}^{K \times D}\) and bias
  \(b \in \mathbb{R}^K\) (shared across positions). For each token position
  \(i = 1,\ldots,N\) we compute

$$
z_i = W h_i + b,
$$

  and apply a softmax to \(z_i\) to obtain a probability distribution over
  \(K\) possible labels for that token (e.g. `PERSON`, `LOC`,
  `COLOR`, etc.).

  During training, each token in the input sequence has a ground-truth
  label, and we sum the cross-entropy loss over all token positions
  (optionally skipping special tokens such as padding). At test time, we
  simply pick the most likely label for each position, giving a predicted
  tag sequence aligned with the original input tokens.

During fine-tuning, all parameters, including the new output layer, are
updated using stochastic gradient descent to maximize the log probability of the
correct labels. Finally, instead of a simple classifier head, the encoder’s representations can
also be fed into a more advanced generative model, for example in text-to-image
synthesis systems.

## Sequence-to-sequence transformers

The third family of transformer models combines an encoder with a decoder, as in
the original transformer paper of Vaswani et al. (2017). A typical example is
machine translation, say from English to Dutch. Let:

- \(x_1,\dots,x_M\) = tokens of the **English** sentence 
- \(y_1,\dots,y_N\) = tokens of the **Dutch** sentence 

### Decoder-only transformer (GPT-style)

Here we model a single sequence only, lets say english:

- Input tokens: \(x_1,\dots,x_M\).
- Each \(x_t\) is embedded to a vector \(e_t\) with \(e_t \in \mathbb{R}^D\), using an embedding matrix \(E \in \mathbb{R}^{K \times D}\).
- Masked self-attention processes \((e_1,\dots,e_{t-1})\) to produce a
  hidden state \(h_t\) for position \(t\) (with \(h_t \in \mathbb{R}^D\)).
- A linear+softmax layer turns \(h_t\) into a distribution over the English
  vocabulary:

$$
p(x_t \mid x_1,\dots,x_{t-1}).
$$

  Concretely, a weight matrix \(W^{\text{out}} \in \mathbb{R}^{D \times K}\)
  and bias \(b^{\text{out}} \in \mathbb{R}^K\) map \(h_t\) to logits in
  \(\mathbb{R}^K\), which are then passed through a softmax to get a length-\(K\)
  probability vector. So, for this we only have one sequence and one language. Each token is predicted from the
  previous tokens in that same sequence.

### Encoder–decoder transformer (seq2seq for translation)

Now we truly have *two* sequences:

$$
x_1,\dots,x_M \ (\text{English source}), \qquad
y_1,\dots,y_N \ (\text{Dutch target}).
$$

**Encoder (English side).**

(i) Each English token \(x_m\) is embedded to \(e^{\text{src}}_m\)
  (with \(e^{\text{src}}_m \in \mathbb{R}^D\), using a source embedding matrix
  \(E^{\text{src}} \in \mathbb{R}^{K_{\text{src}} \times D}\)).
  
(ii) Bidirectional self-attention over all \(e^{\text{src}}_1,\dots,e^{\text{src}}_M\)
  produces encoder states \(z_1,\dots,z_M\)
  (each \(z_m \in \mathbb{R}^D\)).
  Each \(z_m\) summarizes information about the *whole* English
  sentence, but is still tied to position \(m\).

**Decoder (Dutch side).**

(i) We have a target (Dutch) sequence with tokens \(y_1,\dots,y_N\). During training
  we feed the decoder the *shifted* input sequence
  \((\langle\text{start}\rangle, y_1,\dots,y_{N-1})\) and train it to predict
  \((y_1,\dots,y_N)\).
  
(ii) Each token in this decoder input is embedded
  to \(e^{\text{tgt}}_n\) (with \(e^{\text{tgt}}_n \in \mathbb{R}^D\), via a target embedding matrix \(E^{\text{tgt}} \in \mathbb{R}^{K_{\text{tgt}} \times D}\)).
  
(iii) Masked self-attention over these target embeddings produces intermediate
  states \(\hat{h}_n\), where each \(\hat{h}_n\) can only attend to earlier positions in
  the decoder input, i.e. to \(y_1,\dots,y_{n-1}\) (no peeking at future Dutch
  tokens). Each \(\hat{h}_n \in \mathbb{R}^D\).
  
(iv) Now we use Cross-attention where the key and query comes from different datasets or sequences (here English and Dutch). In this for each position \(n\) we:
  
  - use \(\hat{h}_n\) as a *query* \(q_n\),
  - use all encoder states \(z_1,\dots,z_M\) as *keys* and *values*.

  Concretely, we apply learned projection matrices
  \(W^Q, W^K, W^V \in \mathbb{R}^{D \times D}\) to obtain

$$
  q_n = \hat{h}_n W^Q,\quad
  k_m = z_m W^K,\quad
  v_m = z_m W^V,
$$

  where \(q_n, k_m, v_m \in \mathbb{R}^D\). The attention scores are first computed from the *query* \(q_n\) and each
  *key* \(k_m\):

$$
  s_{n,m} = q_n^\top k_m,
$$

  so \(s_{n,m}\) is a scalar, and the score matrix \(S = [s_{n,m}]\) has shape
  \(\mathbb{R}^{N \times M}\). We then turn these scores into attention weights by applying a softmax over
  \(m\):

$$
  \alpha_{n,m}
  = \frac{\exp(s_{n,m})}{\sum_{j=1}^M \exp(s_{n,j})}
  \;\;\propto\;\; \exp(q_n^\top k_m).
$$

  Each \(\alpha_{n,m}\) is a scalar, and for fixed \(n\) the vector
  \((\alpha_{n,1},\dots,\alpha_{n,M})\) lies in \(\mathbb{R}^M\) and sums to \(1\). Finally, the cross-attention output at position \(n\) is a weighted average of
  the *value* vectors \(v_m\):

$$
  c_n = \sum_{m=1}^M \alpha_{n,m} v_m,
$$

  so \(c_n \in \mathbb{R}^D\), and the representation at position \(n\) can directly “look at” *any*
  English position \(m\) via its weight \(\alpha_{n,m}\).

(v) \(c_n\) is then combined with \(\hat{h}_n\) using the usual transformer
  block structure: first a residual (skip) connection, then layer
  normalization, and then a feed-forward network. Concretely, we can write

$$
  u_n = \mathrm{LayerNorm}(\hat{h}_n + c_n), \qquad
  h_n = \mathrm{FFN}(u_n),
$$

  where \(\mathrm{FFN}\) is a small position-wise MLP that maps
  \(\mathbb{R}^D \to \mathbb{R}^D\). Both \(u_n\) and \(h_n\) are in
  \(\mathbb{R}^D\). The final state
  \(h_n\) then goes through a linear+softmax layer to produce

$$
  p(y_n \mid y_1,\dots,y_{n-1}, x_1,\dots,x_M).
$$

  Here a projection \(W^{\text{tgt}} \in \mathbb{R}^{D \times K_{\text{tgt}}}\) and bias
  \(b^{\text{tgt}} \in \mathbb{R}^{K_{\text{tgt}}}\) map \(h_n\) to logits in
  \(\mathbb{R}^{K_{\text{tgt}}}\), followed by a softmax over the \(K_{\text{tgt}}\) target tokens.

(vi) The encoder and decoder are *trained together, end-to-end*. A
  training pair \((x_{1:M}, y_{1:N})\) consists of a source sequence \(x_{1:M}\) (e.g. an English sentence) and
  its corresponding target sequence \(y_{1:N}\) (e.g. the Dutch translation).
  For each training pair, we:
  
  - run the encoder on the *entire* source sequence \(x_{1:M}\),
  - feed the shifted target sequence
    \((\langle\text{start}\rangle, y_1,\dots,y_{N-1})\) into the decoder,
  - obtain a predicted distribution for every position in the target
    sequence and compute a cross-entropy loss at each step \(n\) for \(y_n\).

  In this way, every part of the target sequence contributes to the loss, and
  gradients update all parameters (encoder, decoder, and
  cross-attention) jointly. 

**Key points:**

- We **still** never allow \(y_n\) to see future \(y_{n+1},y_{n+2},\dots\),
  so there is no data leakage.
- What **changes** compared to the decoder-only model is that each
  target token \(y_n\) can now attend to *all* encoder states
  \(z_1,\dots,z_M\), i.e. to the entire English sentence \(x_1,\dots,x_M\),
  via cross-attention.

Intuitively, this is like a user sending their query to a different streaming
service: the service compares the query with its own library of key vectors and
returns the best-matching movie as the value vector. When we wire the encoder and decoder together in this way we obtain the classic
sequence-to-sequence transformer architecture. The model is trained in a supervised way using *sentence pairs*: for each
English input sentence \(x_1,\dots,x_M\) we provide the corresponding Dutch output
sentence \(y_1,\dots,y_N\), and the network learns to map the full source sentence
to its correct translated target sentence.


## References

- Bishop, C. M., & Bishop, H. (2023). Transformers. In Deep Learning: Foundations and Concepts (pp. 357-406). Cham: Springer International Publishing.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... and Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
