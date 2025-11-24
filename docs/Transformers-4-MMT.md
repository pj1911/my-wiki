# Multimodal Transformers

## Introduction

Transformers were first introduced as an alternative to recurrent networks for handling sequential language data. Today, they are used across almost all areas of deep learning.

A key reason is that transformers are very general—they make only weak assumptions about the structure of the input. This contrasts with convolutional networks, which strongly assume that important patterns are local and behave similarly when shifted across the input (equivariance and locality).

- **Locality** means that each neuron in a convolutional layer only looks at a small neighborhood (a local patch) of the input, not the whole image or signal at once. We are building in the belief that useful features like edges, corners, or textures can be detected from nearby pixels.
- **Equivariance** means that if we shift the input (for example, move an object a few pixels to the left), the feature maps produced by the convolution also shift in the same way. The pattern is recognized no matter where it appears, and the response simply moves along with it. This is a very strong built-in assumption about how patterns behave across space, and it is one of the reasons convolutions work so well on images but are less general than transformers.

Because of this generality, transformers have become state-of-the-art on many data types, including text, images, video, point clouds, and audio. Within each of these domains, they are used for both discriminative tasks (such as classification) and generative tasks (such as synthesis). Interestingly, the basic transformer layer architecture has stayed almost the same over time and across applications. Most of the innovation needed to move from pure language to other domains has focused instead on how we *represent and encode* the inputs and outputs so that the transformer can work with them.

Having a single architecture that can process many kinds of data also makes *multimodal* applications much easier. Here, “multimodal” means we combine two or more types of data in the inputs, the outputs, or both. For example, we can generate an image from a text prompt, or we can build a robot that fuses information from cameras, radar, and microphones.

The key takeaway is simple: if we can tokenize the inputs and later decode the output tokens back into a useful form, there is a good chance that a transformer can be applied.

## Vision transformers

Transformers also work very well for vision and now reach state-of-the-art results on many image tasks. When we use a standard transformer encoder on images, we call it a *vision transformer* (ViT). To use a transformer, we must first turn an image into a sequence of tokens. The simplest idea is to treat each pixel as one token after a linear projection, but this is usually impossible in practice because in this case the memory cost of a transformer grows roughly with the *square* of the number of tokens/pixels. Instead, ViTs almost always use *patch tokens*. Assume an image

$$
\mathbf{x} \in \mathbb{R}^{H \times W \times C},
$$

where \(H\) and \(W\) are height and width in pixels and \(C\) is the number of channels (typically \(C=3\) for RGB). In a ViT we make a different design choice: *one token = one image patch*. So pixels are no longer tokens. To achieve this we cut the image into non-overlapping \(P \times P\) patches (e.g. \(P = 16\)). Each patch contains \(P \times P \times C\) pixel values, which we reshape into a vector of length \(P^{2}C\). This raw patch vector is then mapped, via a linear layer, to a \(D\)-dimensional embedding, that \(D\)-dimensional vector is *one token*. Doing this for all patches gives

$$
\mathbf{x}_p \in \mathbb{R}^{N \times (P^{2} C)},
$$

before the linear projection, where

$$
N = \frac{HWC}{P^{2}C}
= \frac{HW}{P^{2}}
$$

is the number of patches, and therefore the number of tokens.
Each patch (and thus each token) contains \(P^{2}\) pixels in space and \(P^{2}C\) pixel values in total (because there are \(C\) channels). After projection, each token is a \(D\)-dimensional vector fed into the transformer.

Another way to create tokens is to first pass the image through a small convolutional neural network (CNN). The CNN down-samples the spatial resolution, and we then treat each spatial location of the final feature map as one token. For example, a typical ResNet18 encoder reduces height and width each by a factor of \(8\), so we obtain \(64\) times fewer tokens than raw pixels.

**Positional embeddings.** We also need to encode where each patch comes from. One option is to build explicit 2D positional embeddings that represent the \((x,y)\) location of each patch. In practice, this rarely helps compared to simply learning a positional embedding vector per token index, so learned positional embeddings are more common. Unlike NLP transformers, vision transformers usually assume a fixed number of tokens (for example, always \(14 \times 14\) patches), so these learned embeddings do not need to generalize to different input lengths or image sizes.

Architecturally, a ViT is very different from a CNN. CNNs have strong built-in inductive biases: weight sharing, locality, and approximate translation equivariance. In a ViT, the only real inductive bias is the decision to slice the image into patches, everything else about image geometry must be learned from data. As a result, ViTs typically need more training data than comparable CNNs. The upside is that, because they do not hard-code many assumptions about the input structure, transformers can often reach higher accuracy once enough data and compute are available. This nicely illustrates the trade-off between strong inductive biases and the amount of training data.

## Generative image transformers

In language, transformers shine when used as *autoregressive* generators: they
predict the next token given all previous ones and can synthesize long texts. A
natural question is whether we can do the same for images. Language is intrinsically sequential, so an autoregressive ordering is obvious.
Images, in contrast, have no natural pixel order. Mathematically, however, any
joint distribution over variables \(\mathbf{x}_1,\dots,\mathbf{x}_N\) can be written
as a product of conditionals once we pick *some* ordering:

$$
p(\mathbf{x}_1,\dots,\mathbf{x}_N)
= \prod_{n=1}^N p(\mathbf{x}_n \mid \mathbf{x}_1,\dots,\mathbf{x}_{n-1}).
\tag{1}\label{eq:autoregressive}
$$

This factorization is completely general and does not restrict the form of the
conditionals \(p(\mathbf{x}_n \mid \mathbf{x}_1,\dots,\mathbf{x}_{n-1})\).

For images, lets assume we can choose \(\mathbf{x}_n\) to be the RGB vector of the \(n\)-th pixel.
We then need an ordering of pixels. A common choice is a *raster scan*
(left-to-right, top-to-bottom). Generating an image autoregressively means
sampling each pixel in this raster scan order using
the above equation. Autoregressive image models existed well before transformers. PixelCNN and
PixelRNN, for example, used specially masked convolutions so that each pixel only
depends on earlier pixels in the raster order.

Real-valued (continuous) image representations work very well for *discriminative* tasks such as
classification: a CNN can map real-valued pixels to real-valued features and then to class scores with no
problem. For *generation*, however, if we model the conditionals with continuous distributions such as
Gaussians and train by maximum likelihood, the model is encouraged to predict the *average* of all
plausible pixel values, which often leads to smooth, blurry images. Discrete representations avoid this by
treating each pixel or patch as choosing from a finite set of codes and modelling a categorical distribution
(over these codes) with a softmax. In this case a conditional
\(p(\mathbf{x}_n \mid \mathbf{x}_1,\dots,\mathbf{x}_{n-1})\) can assign high probability to both “black” and
“white” for a pixel, rather than collapsing to “grey”, so sharp, multimodal structure is captured much
more naturally and samples are typically higher quality.

However, working directly with *discrete pixels* is still difficult. A single
colour pixel has 8 bits for each of the three RGB channels, so each channel can
take \(2^8 = 256\) values. The total number of possible colours per pixel is
\(256^3 = 2^{24} \approx 16\text{M}\), so using a separate softmax over all options for
every pixel is computationally impractical.
A popular fix is
*vector quantization*, which we can view as learned compression.

### Vector quantization  
Assume our dataset can be written as a matrix
\(\mathbf{X} \in \mathbb{R}^{N \times D}\), where each row is a data vector
\(\mathbf{x}_1,\dots,\mathbf{x}_N \in \mathbb{R}^D\) (e.g. pixels). We also have
a set of \(K\) codebook vectors
\(\mathcal{C} = \{\mathbf{c}_1,\dots,\mathbf{c}_K\} \subset \mathbb{R}^D\),
with \(K \ll D\). We approximate each data vector by its nearest codebook vector,
usually in Euclidean distance:

$$
\mathbf{x}_n \;\rightarrow\;
\arg\min_{\mathbf{c}_k \in \mathcal{C}} \|\mathbf{x}_n - \mathbf{c}_k\|^2.
\tag{2}\label{eq:vq}
$$

Because there are only \(K\) codebook vectors, we can represent each
\(\mathbf{x}_n\) by a \(K\)-dimensional one-hot code, so the whole dataset becomes
a matrix of codes \(\mathbf{Z} \in \{0,1\}^{N \times K}\) (or, equivalently, an
index vector in \(\{1,\dots,K\}^N\)). By choosing \(K\), we control a trade-off:
larger \(K\) gives a more faithful representation, smaller \(K\) gives stronger
compression. We can now map all pixels into this lower-dimensional codebook
space, train an autoregressive transformer to generate sequences of code
indices, and finally map these indices back to image pixels by replacing index
\(k\) with its codebook vector \(\mathbf{c}_k\). This reconstruction is only
approximate: we generally cannot recover the exact original \(\mathbf{x}_n\), only
its nearest codebook vector \(\mathbf{c}_k\). In practice, we pick \(K\) (and learn
the codebook) so that this quantization error is small enough that the generated
images still look sharp and realistic.

**ImageGPT** was one of the first autoregressive transformers for images. It clusters
the colour space with \(K\)-means and treats each pixel as belonging to one of the
resulting \(K\) RGB codebook vectors. The one-hot codes act as discrete tokens,
just like words in language models, and the transformer is trained with a
next-token prediction loss. This objective gives strong image representations
that can be fine-tuned for downstream tasks, mirroring language modelling.

As in vision transformers, it is more efficient to use *patches* as tokens.
Fewer tokens make high-resolution images feasible. We still want discrete tokens to capture multimodal conditionals, but now the
*dimensionality explodes*: the number of possible patches grows
exponentially with the number of pixels in a patch. If we take \(16\times 16\)
patches with just two pixel values (black/white), there are

$$
16 \times 16 = 256 \quad\Rightarrow\quad \text{number of patches}
= 2^{256} \approx 1.16 \times 10^{77},
$$

since each of the \(256\) pixels can independently be black or white.

So we again turn to vector quantization, now applied to patches. We learn a
codebook of patch vectors from data, using methods like \(K\)-means, fully
convolutional networks, or even vision transformers. A difficulty is that the
quantization step (the nearest-codebook lookup) is not differentiable. In
practice, we use the *straight-through estimator*: during backpropagation we
simply copy gradients through the non-differentiable step as if it were the
identity function. Finally, the same idea extends naturally from images to videos. We treat a video
as one long sequence of vector-quantized tokens (over space and time) and train
an autoregressive transformer over this sequence to generate videos frame by
frame.

## Audio data

Transformers can also process audio. Raw sound is usually stored as a
*waveform*: a sequence of air-pressure samples over time. Instead of using
this directly, we usually convert it to a *mel spectrogram*, a matrix whose
columns are time steps and rows are frequency bands on the mel scale, designed
so equal steps roughly match equal perceived pitch changes.

A core task is audio classification, where short clips are assigned labels such
as *car*, *animal*, or *laughter*. A common benchmark is the
*AudioSet* dataset. Before transformers, the best systems treated mel
spectrograms as images and used CNNs. CNNs capture local patterns well but
struggle with long-range temporal dependencies, which often matter for audio.

Now, transformers are increasingly used instead. A transformer encoder with the
same architecture as in language or vision can classify audio. We view the mel
spectrogram as an image, split it into patches (optionally overlapping), and
flatten each patch into a 1D vector. Each patch
becomes a token, we add positional encodings, prepend a special `<class>`
token, and feed all tokens to the transformer encoder. The final output at the
`<class>` position goes through a linear layer and a softmax, and the whole
model is trained end-to-end with a cross-entropy loss.

## Text-to-speech

Classification is only one success story for transformers in audio. Another is
*text-to-speech* (TTS): generating spoken audio that follows a given text,
often in the voice of a specific speaker. In a traditional TTS system, we record many examples of a single speaker and
train a supervised regression model to map text to a low-level representation of
speech, such as a mel spectrogram. At inference time, we feed in new text, get a
predicted spectrogram, and convert it deterministically back to a waveform.

This setup has several drawbacks. If we predict very low-level units (for
example, phonemes), the model must handle long-range context to make sentences
sound natural. If we instead predict longer segments, the input space becomes
huge and requires impractically large datasets. The approach also does not share
knowledge across speakers, so each new voice needs a lot of data. Finally, TTS
is inherently a generative problem: many different speech signals are valid for
the same text and speaker, while regression tends to average them into bland,
less expressive outputs.

A more modern view treats speech like language and frames TTS as
*conditional language modelling*. We still use transformers, but now the
model predicts the next audio token given previous audio tokens and the input
text. The main design questions become: (1) how to tokenize speech so we can
decode predictions back to audio, and (2) how to condition the model on the
desired speaker’s voice.

First, speech is converted into a sequence of *speech tokens* using vector
quantization. We learn a codebook (dictionary) of audio embeddings, split a
waveform into short frames, and replace each frame by the index of its nearest
codebook vector. During training, the transformer input is a single sequence
built by concatenating (i) the text tokens for the sentence we want to speak and
(ii) a short run of speech tokens taken from a separate sample of the same
speaker. The model is trained to output the speech tokens that correspond to the
full spoken version of the input text. Intuitively, the text tokens tell the
model *what* to say, while the conditioning speech tokens tell it *in
which voice* to say it.

At test time, we provide new text plus a brief speech sample from a new
speaker. The model generates speech tokens conditioned on both the text and this
speaker snippet. Finally, we map the tokens back through the same codebook to
produce a waveform. This lets the system read out arbitrary text in the voice of
a speaker it has only heard for a few seconds.

## Vision and language transformers

So far we have seen how to build discrete tokens for text, audio, and images.
A natural next step is to mix modalities: let the input tokens come from one
modality and the output tokens from another, or even use several modalities on
both sides. In practice, the most studied case is text + vision, but the same
ideas extend to other combinations.

The first ingredient is a large multimodal dataset. For text–image work, the
LAION-400M dataset has played a role similar to ImageNet for image
classification, enabling rapid progress in both text-to-image generation and
image captioning. Text-to-image generation is very close to unconditional image
generation, except that we now *condition* on a text prompt. With transformers, conditioning on text is straightforward. We first encode the
prompt into text tokens and keep them in the context, at every step when the
model predicts the next image token, its attention layers can look back at both
the previously generated image tokens *and* these fixed text tokens, so the
visual details follow the words in the prompt.

We can also view text-to-image as a standard sequence-to-sequence problem, like
machine translation, but with discrete image tokens as the target sequence
instead of words. This motivates using a full encoder–decoder transformer: the
encoder reads the text tokens \(X\), and the decoder outputs the image tokens \(Y\).
Models such as Parti follow this pattern and scale the transformer to tens of
billions of parameters, with performance improving as the model size increases.

Another research line starts from large pre-trained language models and
adapts them so they can also accept visual inputs. These systems usually use
custom modules that map images to continuous feature vectors, which are then
injected into the language model. Because the visual representation is tied to a
specific encoder and feature format, it is awkward to plug in new kinds of data,
such as audio or video, without redesigning this interface. This also makes it
harder to apply the same model to *generate* images, since the model never
sees discrete image tokens that could be decoded back into pixels in a unified
way. Ideally, we would like a single model that can consume and
produce both text and image tokens (and possibly more). The simplest recipe is
to treat everything as one long token sequence and define a joint vocabulary
that is just the union of the text token dictionary and the image token
codebook. The key point is that *all* modalities now share this one
vocabulary, so a single transformer can read and generate mixed streams of
tokens (text, image, audio, …) without any modality-specific heads or
separate architectures.

Models such as CM3 and CM3Leon follow this language-modelling view. They are
trained on HTML pages from the web that contain both text and images, using a
variant of next-token prediction over the mixed token stream. With enough
training data and a scalable architecture, these models become very powerful and
flexible: they can do text-to-image generation, image captioning, image editing,
text completion, and essentially any task a regular language model can handle,
all within a single multimodal transformer.


