# Generative Adversarial networks

## Generative modeling problem

The generative modeling problem (GMP) is the task of learning a probability distribution from data so that we can generate new, realistic samples that look like they came from the real dataset. GANs are a kind of AI algorithm that are designed to solve GMPs. They are especially famous for how good they are at creating realistic, high-resolution images.

### Generative modeling
Generative modeling is an unsupervised approach where we observe samples \(\mathbf{x}\) from unknown distribution (\(\mathbf{x} \sim p_{\text{data}}(\mathbf{x})\)) and learn a model
\(p_{\text{model}}(\mathbf{x})\) that matches this distribution. We choose a
parametric form \(p_{\text{model}}(\mathbf{x}; \theta)\) and fit \(\theta\) so
\(p_{\text{model}}\) resembles \(p_{\text{data}}\), typically by maximum
likelihood, i.e., minimizing
\(\mathrm{KL}\big(p_{\text{data}} \,\|\, p_{\text{model}}\big)\).

### Common issues
Explicit density models work nicely in classic statistics, where we use simple
distributions over a few variables. In modern deep learning, however, we often
use complex neural networks, and their exact density can be intractable. People
have mostly tried two fixes: (1) design models with tractable densities, or
(2) use approximate methods to learn intractable ones. Both are hard and still
struggle on tasks like generating realistic high-resolution images.

### Fix: Learn sampling procedure directly
An alternative to these explicit density models is to skip the tractable density entirely and learn only a
tractable sampling procedure. These are called *implicit generative models* and
GANs belong to this family. Before GANs, the leading deep implicit model was the
generative stochastic network, which produced approximate samples via a
Markov chain. GANs were proposed to instead generate high-quality samples in
a *single* step, avoiding the incremental and approximate nature of
Markov-chain sampling.

## Generative adversarial networks

GANs are built in the game-theory sense, a game between two models,
usually neural networks. One of them, the *generator*, implicitly defines
\(p_{\text{model}}(\mathbf{x})\). In general, it cannot compute this density, but
it *can* draw samples from it. The generator starts from a simple prior \(p(\mathbf{z})\) over a latent vector
\(\mathbf{z}\) (for example, a multivariate Gaussian or a uniform distribution
over a hypercube). A sample \(\mathbf{z} \sim p(\mathbf{z})\) is just noise. The
generator is a function \(G(\mathbf{z}; \theta_G)\) that learns to transform this
noise into realistic samples, with \(\theta_G\) representing its learnable
parameters or “strategy’’ in the game.

The other player is the *discriminator*. It looks at a sample
\(\mathbf{x}\) and outputs a score \(D(\mathbf{x}; \theta_D)\) estimating whether
\(\mathbf{x}\) is real (from the training data) or fake (from
\(p_{\text{model}}\), via the generator). In the original GAN, this score is the
probability that \(\mathbf{x}\) is real, assuming real and fake examples are shown
equally often.

Each player has its own cost: \(J_G(\theta_G, \theta_D)\) for the generator and
\(J_D(\theta_G, \theta_D)\) for the discriminator, and each tries to minimize
its own cost. The discriminator’s cost pushes it to classify correctly and the
generator’s cost pushes it to make fake samples that the discriminator classifies as real.

### Loss for Generator and Discriminator
In the original GAN, the discriminator sees two kinds of examples:
real data \(\mathbf{x} \sim p_{\text{data}}\) with label 1, and generated data
\(G(\mathbf{z})\) with label 0. Its loss is just the usual binary
cross-entropy:

$$
J_D(\theta_D, \theta_G)
= - \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \big[\log D(\mathbf{x})\big]
  - \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})} \big[\log \big(1 - D(G(\mathbf{z}))\big)\big].
$$

The first term says “real samples should have \(D(\mathbf{x})\) close to 1,”
and the second says “fake samples should have \(D(G(\mathbf{z}))\) close to 0.”
So \(D\) is trained exactly like a standard real-vs-fake classifier.
For the generator, two options are usually proposed:

- **Minimax GAN (M-GAN):** \(J_G = -J_D\), giving a clean minimax game.
    This makes GAN training a standard zero-sum game: whenever the
    discriminator gets better, the generator “loses,” and vice versa., and at equilibrium this
    setup corresponds to minimizing a well-defined divergence between
    \(p_{\text{data}}\) and \(p_{\text{model}}\).

- **Non-saturating GAN (NS-GAN):**
In the original minimax version, the generator minimizes

$$
J_G^{\text{minimax}}(\theta_G, \theta_D)
= \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}
  \big[\log\big(1 - D(G(\mathbf{z}))\big)\big],
$$

which gives very small gradients when \(D(G(\mathbf{z}))\) is close to 0.

In NS-GAN, we “flip the labels’’ for the generator: it acts as if its fake
samples were real and tries to push \(D(G(\mathbf{z}))\) toward 1. Its loss is

$$
J_G^{\text{NS}}(\theta_G, \theta_D)
= - \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}
  \big[\log D(G(\mathbf{z}))\big],
$$

the negative log-likelihood of the “real’’ label for fake samples. This gives
stronger gradients when the discriminator is confident (i.e., \(D(G(\mathbf{z}))\)
is small), so the generator keeps learning instead of getting stuck.

NS-GAN is often preferred in practice because it helps avoid gradient
saturation during training.

We can think of GANs like counterfeiters and police. The generator is the
counterfeiter, making fake money and the discriminator is the police, trying to
catch fakes while letting real money through. As they compete, the fakes get
better and better until, in the ideal case, the police can no longer tell real
from fake. The twist is that, in GANs, the generator learns from the
discriminator’s gradient, as if the counterfeiters had a mole inside the police
force explaining exactly how they spot fakes.

### How to train GANs
GANs use game-theoretic ideas in a challenging setting: the losses are
non-convex, and both actions and policies live in continuous, high-dimensional
spaces (whether we view an action as choosing parameters \(\theta_G\) or as
producing a sample \(\mathbf{x}\)). The learning goal is to reach a
*local Nash equilibrium*: a point where each player’s loss is at a local
minimum with respect to its own parameters. In such a state, with only small
(local) changes and holding the other player fixed, no player can further
reduce its loss.

The most common way to train a GAN is simple: use a gradient-based optimizer
to update both players’ parameters in turn, each trying to reduce its own
loss. When this works well, the trained generator can produce very realistic
samples, even for complex datasets with high-resolution images.
A high-level reason GANs can be so effective is that they avoid many of the
approximations used in other generative models. We never have to approximate
an intractable density, instead, we directly train the generator to fool the
discriminator. The main sources of error are then just statistical (finite
data) and optimization-related (not reaching the ideal equilibrium), rather
than additional approximation errors from Markov chains, variational bounds,
and so on.

### Convergence of GANs

Convergence for a GAN means that training settles
into a stable state: the generator’s samples and the discriminator’s
predictions stop changing (up to small noise), and neither player can improve
its loss by making a small change in its parameters. The original GAN paper gave two key (idealized) results regarding this:

1. In the space of all possible density functions \(p_{\text{model}}\) and
        discriminators \(D\), there is only *one* local Nash equilibrium:
        the point where the model matches the data perfectly,
        \(p_{\text{model}} = p_{\text{data}}\).
2. If we had an ideal optimizer that, for any fixed \(p_{\text{model}}\),
        could find the best possible discriminator \(D^\ast\), then the
        following loop would converge to that equilibrium:
    1. fix \(p_{\text{model}}\) and optimize \(D\) all the way to
                \(D^\ast\);
    2. then take a small gradient step on \(p_{\text{model}}\) to
                reduce its loss, keeping \(D^\ast\) fixed.

        Repeating these steps would eventually make
        \(p_{\text{model}}\) equal to \(p_{\text{data}}\).

In practice, things are messier. We do not move directly in the space of all
distributions. Instead, both \(p_{\text{model}}\) and \(D\) are neural
networks with finitely many parameters, trained with noisy, alternating
gradient steps. The losses are highly non-convex, and each update of one
player changes the landscape seen by the other. Because of this, training can
oscillate, diverge, or collapse to poor solutions rather than neatly settling
to the nice equilibrium guaranteed in the idealized theory.

## References
- Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2020). Generative adversarial networks. Communications of the ACM, 63(11), 139-144.
