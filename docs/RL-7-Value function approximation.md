## Motivation

Reinforcement learning is often used in settings where the state space is enormous (or even continuous). For instance, backgammon has about \(10^{20}\) states, computer Go has about \(10^{170}\) states, and helicopter control lives in a continuous state space. In such problems, we would like model-free methods to still work well for both prediction and control. However, the standard way we have represented value functions so far with a lookup table with one entry per state \(V(s)\), or one entry per state--action pair \(Q(s,a)\), does not scale: there are simply too many states and/or actions to store in memory, and learning each value independently is too slow.

## Solution: function approximation

To handle large (or continuous) state spaces, we stop storing one number per state (or state--action pair) and instead predict values with a parameter vector \(w\):

$$
\hat{v}(s,w) \approx v_{\pi}(s),
\qquad
\hat{q}(s,a,w) \approx q_{\pi}(s,a).
$$

Here \(w\) are the model's internal parameters (e.g., weights of a linear model or a neural network). A value function is always defined relative to a policy \(\pi\). So in prediction, we usually keep \(\pi\) fixed and learn \(\hat{v}(s,w)\approx v_{\pi}(s)\) (or \(\hat{q}(s,a,w)\approx q_{\pi}(s,a)\)). In control, the policy typically changes as we improve it, or we may aim directly for the optimal value functions:

$$
\hat{v}(s,w) \approx v_*(s),
\qquad
\hat{q}(s,a,w) \approx q_*(s,a).
$$

Function approximation also enables generalization because many states share the same parameters \(w\): updating \(w\) using experience from some states can also change the predicted values of other, possibly unseen, states. We update \(w\) using the same ideas as before (Monte Carlo or Temporal-Difference learning), but instead of editing a table entry, we adjust \(w\) so that \(\hat{v}(s,w)\) or \(\hat{q}(s,a,w)\) moves toward a chosen target (an MC return or a TD target).

### Types of value function approximation

A value-function approximator maps inputs (a state, and sometimes an action) to value estimates. The common setups are: **(1) state-value function approximation**, where the input is \(s\) and the output is a single number \(\hat{v}(s,w)\), **(2) state-action value function approximation for a single action**, where the input is \((s,a)\) and the output is one number \(\hat{q}(s,a,w)\), useful when we only care about the value of a specific action at a specific state and **(3) state-action values for all actions**, where the input is \(s\) and the output is a vector \(\big(\hat{q}(s,a_1,w),\ldots,\hat{q}(s,a_m,w)\big)\), which is especially convenient for discrete action spaces because we get all action-values in one forward pass.

### Models for function approximation

Many models can approximate value functions, including linear feature combinations, neural networks, decision trees, nearest-neighbour methods, Fourier/wavelet bases, etc. In this chapter we focus on differentiable approximators (like linear models and neural networks). Differentiability matters because it gives us gradients: it tells us how the prediction changes when we change the parameters \(w\), which is exactly what we need for gradient-based learning.

One extra wrinkle in reinforcement learning is that the data are usually not i.i.d. They come from trajectories, so consecutive samples are temporally correlated. In control, things are even less stable: as the policy improves, the data distribution can shift over time. So our training methods must cope with non-i.i.d. and non-stationary data.

## Learning methods for value function approximation

When learning with function approximation, updates are usually done in one of two ways: batch methods and incremental methods. Batch methods collect many samples and then update using the whole dataset (or large chunks of it). Incremental methods update continuously, using one sample (or a small mini-batch) at a time. Reinforcement learning most often uses the incremental style, since data arrive sequentially from interaction.

### Gradient descent and stochastic gradient descent (SGD)

To learn the parameter \(w\), we need a systematic way to improve our approximation. The standard approach is to define an objective function that measures how wrong our current predictions are, and then adjust \(w\) to reduce that error. When the objective is differentiable, gradient-based methods give a simple and effective update rule.  
Let \(J(w)\) be a differentiable objective function of the parameter vector \(w\).  
Its gradient is the vector of partial derivatives

$$
\nabla_w J(w) =
\begin{pmatrix}
\frac{\partial J(w)}{\partial w_1}\\
\vdots\\
\frac{\partial J(w)}{\partial w_n}
\end{pmatrix}.
$$

To reduce \(J(w)\), we move parameters in the direction of the negative gradient:

$$
\Delta w = -\frac{1}{2}\alpha \nabla_w J(w),
$$

where \(\alpha>0\) is the step size.

### Value function approximation via SGD

We want parameters \(w\) that minimize the mean-squared error between the approximation \(\hat{v}(s,w)\) and the true value \(v_\pi(s)\):

$$
J(w) = \mathbb{E}_{\pi}\!\left[\left(v_\pi(S) - \hat{v}(S,w)\right)^2\right].
$$

Gradient descent gives

$$
\Delta w
= -\frac{1}{2}\alpha \nabla_w J(w)
= \alpha\, \mathbb{E}_{\pi}\!\left[\left(v_\pi(S) - \hat{v}(S,w)\right)\nabla_w \hat{v}(S,w)\right].
$$

### Incremental (stochastic) update

Computing the expectation in the gradient update is usually infeasible, so we approximate it by sampling. Concretely, we generate a trajectory by following policy \(\pi\), take the current visited state \(S\), and use it as a sample from the state distribution induced by \(\pi\). Replacing the expectation with this single sample gives the incremental (stochastic) update:

$$
\Delta w
= \alpha \left(v_\pi(S) - \hat{v}(S,w)\right)\nabla_w \hat{v}(S,w).
$$

Intuitively, we adjust \(w\) to reduce the current prediction error \(\big(v_\pi(S)-\hat{v}(S,w)\big)\), scaled by the sensitivity of the predictor to its parameters, \(\nabla_w \hat{v}(S,w)\). This is reasonable because the full-gradient update is an average over states:

$$
\alpha\,\mathbb{E}_{\pi}\!\left[\left(v_\pi(S) - \hat{v}(S,w)\right)\nabla_w \hat{v}(S,w)\right].
$$

If we generate experience by following \(\pi\), then each visited state \(S\) can be viewed as a sample from the state distribution induced by \(\pi\). Then SGD replaces this expectation with a single sampled term,

$$
\alpha\left(v_\pi(S)-\hat{v}(S,w)\right)\nabla_w \hat{v}(S,w),
$$

so each step uses only the current state from the trajectory. Individual updates are noisy because \(S\) changes from step to step, and some states may appear more often than others. But because the samples come from the same distribution used in the expectation, the sampled update is unbiased: if we average these incremental updates over many time steps, we recover the expected (full-gradient) update direction. This is why SGD still minimizes \(J(w)\) in expectation, while avoiding the cost of computing the full average at every step.

### Linear value function approximation

Instead of working with the raw state, we describe each state using a feature vector

$$
x(S)=
\begin{pmatrix}
x_1(S)\\
\vdots\\
x_n(S)
\end{pmatrix}.
$$

We can think of features as measurements that summarize what matters about each state \(S\), such as a robot's distance to landmarks, indicators from the stock market, or patterns of pieces in chess. Features can make learning practical, but they may also discard information about the original state. For the remaining of this chapter, we assume we have a reasonably good feature representation.

### Linear value function approximation

A simple and widely used differentiable approximator is a linear model:

$$
\hat{v}(S,w)=x(S)^\top w=\sum_{j=1}^n x_j(S)\,w_j.
$$

Using mean-squared error,

$$
J(w)=\mathbb{E}_{\pi}\!\left[\left(v_\pi(S)-x(S)^\top w\right)^2\right],
$$

the objective is quadratic in \(w\). This is nice: for linear approximation, SGD converges (under standard conditions) to a global minimizer of \(J(w)\). Then the gradient is especially clean:

$$
\nabla_w \hat{v}(S,w)=x(S),
$$

so the incremental update becomes

$$
\Delta w=\alpha\left(v_\pi(S)-\hat{v}(S,w)\right)x(S),
\qquad\text{or}\qquad
\Delta w_j=\alpha\left(v_\pi(S)-\hat{v}(S,w)\right)x_j(S).
$$

### Interpretation of the update

Each step has the intuitive form

$$
\text{update}=\text{step size}\times\text{prediction error}\times\text{feature value}.
$$

If a feature is inactive (\(x_j(S)=0\)), its weight does not change. If a feature has large magnitude, it drives a larger update. Even when the linear model cannot represent \(v_\pi\) perfectly, good features can still capture enough structure for accurate and useful predictions.

### Table lookup as a special case

A lookup table is just linear approximation with one-hot features. If the finite state space is \(\{s_1,\dots,s_n\}\), define

$$
x^{\text{table}}(S)=
\begin{pmatrix}
\mathbb{I}(S=s_1)\\
\vdots\\
\mathbb{I}(S=s_n)
\end{pmatrix},
$$

where \(\mathbb{I}(\cdot)\) is \(1\) if its argument is true and \(0\) otherwise. Then

$$
\hat{v}(S,w)=\big(x^{\text{table}}(S)\big)^\top w =
\sum_{i=1}^n \mathbb{I}(S=s_i)\,w_i = w_i \quad \text{if } S=s_i.
$$

In this case we create one feature per state. So the model simply picks the single parameter tied to the current state, which is exactly what a lookup table does.

## Incremental prediction with value function approximation

Till now, we assumed, the objective used the true value \(v_\pi(s)\) as if an oracle provided it. But, in reinforcement learning we generally do not have such information, we only see rewards (and next states). The practical fix is simple: replace \(v_\pi(S_t)\) with a target computed from experience, and do an incremental SGD update:

$$
\Delta w
= \alpha\left(\text{target}_t-\hat{v}(S_t,w)\right)\nabla_w \hat{v}(S_t,w).
$$

The only thing that changes across methods is how we choose \(\text{target}_t\):

- **Monte Carlo (MC):** use the return \(G_t\),

  $$
  \Delta w
  = \alpha\left(G_t-\hat{v}(S_t,w)\right)\nabla_w \hat{v}(S_t,w).
  $$

- **TD(0):** use the one-step TD target \(R_{t+1}+\gamma \hat{v}(S_{t+1},w)\),

  $$
  \Delta w
  = \alpha\left(R_{t+1}+\gamma \hat{v}(S_{t+1},w)-\hat{v}(S_t,w)\right)\nabla_w \hat{v}(S_t,w).
  $$

- **TD(\(\lambda\)):** use the \(\lambda\)-return \(G_t^\lambda\),

  $$
  \Delta w
  = \alpha\left(G_t^\lambda-\hat{v}(S_t,w)\right)\nabla_w \hat{v}(S_t,w).
  $$

### Monte Carlo with value function approximation

The Monte Carlo return \(G_t\) is a noisy but unbiased sample of the true value \(v_\pi(S_t)\) (its expectation equals \(v_\pi(S_t)\)). This makes MC prediction with function approximation look a lot like supervised learning: from each episode we get training pairs

$$
\langle S_1,G_1\rangle,\ \langle S_2,G_2\rangle,\ \ldots,\ \langle S_T,G_T\rangle.
$$

We then apply the generic incremental update with \(\text{target}_t=G_t\):

$$
\Delta w=\alpha\left(G_t-\hat{v}(S_t,w)\right)\nabla_w \hat{v}(S_t,w).
$$

In this setting, we apply the update once per time step \(t\) (for each visited state \(S_t\)), using that step's return \(G_t\) as the target. In an episodic task, \(G_t\) depends on rewards all the way until the end of the episode, so we typically:

- run an episode to termination,
- compute \(G_t\) for \(t=0,1,\dots,T-1\),
- then update \(w\) for each time step (either in forward or backward order):

  $$
  w \leftarrow w + \alpha\left(G_t-\hat{v}(S_t,w)\right)\nabla_w \hat{v}(S_t,w).
  $$

For a linear approximator \(\hat{v}(s,w)=x(s)^\top w\) (so \(\nabla_w \hat{v}(S_t,w)=x(S_t)\)), this becomes

$$
\Delta w=\alpha\left(G_t-\hat{v}(S_t,w)\right)x(S_t).
$$

With non-linear approximators, the objective is generally non-convex, so MC prediction typically converges to a local optimum rather than guaranteeing a global one.

### TD learning with value function approximation

Monte Carlo uses the full return \(G_t\) as a target, which is unbiased but only available after the episode ends. TD methods trade a bit of bias for faster, online learning by building targets from the next reward and the current value estimate. The TD(0) target

$$
R_{t+1}+\gamma \hat{v}(S_{t+1},w)
$$

is a biased estimate of \(v_\pi(S_t)\) because it bootstraps from the current approximation \(\hat{v}(\cdot,w)\). Still, we can think of TD as supervised learning on moving targets, with training pairs like

$$
\langle S_t,\ R_{t+1}+\gamma \hat{v}(S_{t+1},w)\rangle.
$$

Notably, the TD target depends on \(\hat{v}(\cdot,w)\), so the ``label'' changes as \(w\) changes. To derive the TD(0) update, we start from the generic incremental SGD rule

$$
\Delta w=\alpha\left(\text{target}_t-\hat{v}(S_t,w)\right)\nabla_w \hat{v}(S_t,w).
$$

Substituting the value of the TD(0) target into the generic rule gives

$$
\Delta w
=\alpha\left(R_{t+1}+\gamma \hat{v}(S_{t+1},w)-\hat{v}(S_t,w)\right)\nabla_w \hat{v}(S_t,w).
$$

It is common to name the term in parentheses the TD error,

$$
\delta_t \;\triangleq\; R_{t+1}+\gamma \hat{v}(S_{t+1},w)-\hat{v}(S_t,w),
$$

so the update can be written compactly as \(\Delta w=\alpha\,\delta_t\,\nabla_w \hat{v}(S_t,w)\). For a linear model \(\hat{v}(s,w)=x(s)^\top w\) (so \(\nabla_w \hat{v}(S_t,w)=x(S_t)\)), this becomes

$$
\Delta w=\alpha\,\delta_t\,x(S_t),
\qquad
\delta_t=R_{t+1}+\gamma \hat{v}(S_{t+1},w)-\hat{v}(S_t,w).
$$

A useful practical note: unlike MC, TD(0) can update immediately at each time step because its target does not require waiting for the episode to end.

### TD(\(\lambda\)) with value function approximation

Like TD(0), the \(\lambda\)-return \(G_t^\lambda\) bootstraps from current estimates, so it is also a biased target for \(v_\pi(S_t)\). We can still think in supervised-learning terms, using pairs

$$
\langle S_1,G_1^\lambda\rangle,\ \langle S_2,G_2^\lambda\rangle,\ \ldots,\ \langle S_{T-1},G_{T-1}^\lambda\rangle.
$$

**Forward view (linear TD(\(\lambda\))).**

Using the generic SGD rule with \(\text{target}_t=G_t^\lambda\) gives

$$
\Delta w
=\alpha\left(G_t^\lambda-\hat{v}(S_t,w)\right)\nabla_w \hat{v}(S_t,w).
$$

For a linear approximator \(\hat{v}(s,w)=x(s)^\top w\) (so \(\nabla_w \hat{v}(S_t,w)=x(S_t)\)),

$$
\Delta w=\alpha\left(G_t^\lambda-\hat{v}(S_t,w)\right)x(S_t).
$$

**Backward view (linear TD(\(\lambda\))).**

The forward view is conceptually clean, but it still refers to \(G_t^\lambda\), which is defined using future rewards. The backward view implements the same idea online by accumulating credit with an eligibility trace. We define the TD error

$$
\delta_t=R_{t+1}+\gamma \hat{v}(S_{t+1},w)-\hat{v}(S_t,w),
$$

and the eligibility trace

$$
E_t=\gamma\lambda E_{t-1}+x(S_t),
\qquad E_{-1}=0.
$$

Then the update is

$$
\Delta w=\alpha\,\delta_t\,E_t.
$$

## Control with value function approximation

In control, the goal is not just to predict values under a fixed policy, but to improve the policy. A common pattern is to alternate between:

- **Policy evaluation:** learn an approximation to the action-value function,

  $$
  \hat{q}(\cdot,\cdot,w)\approx q_\pi,
  $$

- **Policy improvement:** make \(\pi\) more greedy with respect to \(\hat{q}\) (often via \(\epsilon\)-greedy).

### Action-value function approximation

For evaluation, we would ideally choose \(w\) to minimize the mean-squared error between \(\hat{q}(S,A,w)\) and the true \(q_\pi(S,A)\):

$$
J(w)=\mathbb{E}_{\pi}\!\left[\left(q_\pi(S,A)-\hat{q}(S,A,w)\right)^2\right].
$$

Applying gradient descent gives

$$
\Delta w
=-\frac{1}{2}\alpha \nabla_w J(w)
=\alpha\,\mathbb{E}_{\pi}\!\left[\left(q_\pi(S,A)-\hat{q}(S,A,w)\right)\nabla_w \hat{q}(S,A,w)\right].
$$

In practice we use a sampled (incremental) version of this update, replacing the expectation and the unknown \(q_\pi\) with targets estimated from experience (e.g., TD targets).

### Linear action-value function approximation

For action-values, we can describe a state-action pair with a feature vector

$$
x(S,A)=
\begin{pmatrix}
x_1(S,A)\\
\vdots\\
x_n(S,A)
\end{pmatrix},
$$

and use a linear model

$$
\hat{q}(S,A,w)=x(S,A)^\top w=\sum_{j=1}^n x_j(S,A)\,w_j.
$$

The gradient is immediate:

$$
\nabla_w \hat{q}(S,A,w)=x(S,A).
$$

So the incremental SGD update becomes

$$
\Delta w=\alpha\left(q_\pi(S,A)-\hat{q}(S,A,w)\right)x(S,A).
$$

(In practice, \(q_\pi(S,A)\) is unknown and is replaced by a target estimated from experience.)

### Incremental control algorithms with function approximation

Just like in prediction, we do not know the true action-value \(q_\pi(S,A)\), so we learn from targets built from experience. With a differentiable approximator \(\hat{q}(S,A,w)\), the generic incremental update is

$$
\Delta w
=\alpha\left(\text{target}_t-\hat{q}(S_t,A_t,w)\right)\nabla_w \hat{q}(S_t,A_t,w).
$$

Different control algorithms mainly differ in how they choose \(\text{target}_t\):

- **MC control:** use the return \(G_t\),

  $$
  \Delta w
  =\alpha\left(G_t-\hat{q}(S_t,A_t,w)\right)\nabla_w \hat{q}(S_t,A_t,w).
  $$

- **TD(0) / SARSA-style:** bootstrap from the next state--action,

  $$
  \text{target}_t \;\triangleq\; R_{t+1}+\gamma \hat{q}(S_{t+1},A_{t+1},w),
  $$

  giving

  $$
  \Delta w
  =\alpha\left(R_{t+1}+\gamma \hat{q}(S_{t+1},A_{t+1},w)-\hat{q}(S_t,A_t,w)\right)\nabla_w \hat{q}(S_t,A_t,w).
  $$

- **Forward-view TD(\(\lambda\)):** use an action-value \(\lambda\)-return \(q_t^\lambda\),

  $$
  \Delta w
  =\alpha\left(q_t^\lambda-\hat{q}(S_t,A_t,w)\right)\nabla_w \hat{q}(S_t,A_t,w).
  $$

- **Backward-view TD(\(\lambda\)) (eligibility traces):** define the TD error

  $$
  \delta_t
  =R_{t+1}+\gamma \hat{q}(S_{t+1},A_{t+1},w)-\hat{q}(S_t,A_t,w),
  $$

  and an eligibility trace

  $$
  E_t=\gamma\lambda E_{t-1}+\nabla_w \hat{q}(S_t,A_t,w),
  \qquad E_{-1}=0,
  $$

  then update

  $$
  \Delta w=\alpha\,\delta_t\,E_t.
  $$

### Note on continuous actions

With continuous action spaces, policy improvement via \(\epsilon\)-greedy requires

$$
\arg\max_a \hat{q}(s,a,w),
$$

which can itself be a non-trivial optimisation problem. Good feature design can encode prior knowledge and impose helpful structure on \(\hat{q}\), making this optimisation easier (e.g.\ smoother or more convex landscapes).

### Convergence and when bootstrapping can fail

Bootstrapping (using TD targets) often lowers variance and improves sample efficiency, but with function approximation it is not always stable.

### Convergence summary (rule of thumb)

A handy mental model is:

- **Monte Carlo (MC):** usually stable with both linear and non-linear approximation, but can have high variance.
- **On-policy TD + linear approximation:** typically converges under standard conditions (and often works well in practice).
- **TD + non-linear approximation:** no general convergence guarantee; it can diverge.
- **Off-policy TD + function approximation:** can be unstable and may diverge.

So bootstrapping is powerful, but not automatically ``safe.'' In practice, stability is more likely when learning is on-policy, step sizes are chosen carefully, or we use specialised off-policy methods designed for stability.

## Gradient TD learning and it's convergence

With function approximation, it is tempting to think TD is minimizing a single error measure like

$$
J(w)=\mathbb{E}_\pi\!\left[\left(v_\pi(S)-\hat{v}(S,w)\right)^2\right].
$$

For Monte Carlo, the update can be seen as (stochastic) gradient descent on this objective, because the target \(G_t\) is an unbiased sample of \(v_\pi(S_t)\). For TD, however, the target

$$
R_{t+1}+\gamma \hat{v}(S_{t+1},w)
$$

itself depends on \(w\). As a result, the usual TD update

$$
\Delta w=\alpha\left(R_{t+1}+\gamma \hat{v}(S_{t+1},w)-\hat{v}(S_t,w)\right)\nabla_w \hat{v}(S_t,w)
$$

is not, in general, the gradient of a single fixed scalar objective in \(w\) (especially off-policy, and with non-linear approximation). That is a core reason TD can be unstable or even diverge. The TD update is built from the Bellman equation and a bootstrapped target, e.g.

$$
\Delta w \propto \delta_t \nabla_w \hat{v}(S_t,w),
\qquad
\delta_t = R_{t+1}+\gamma \hat{v}(S_{t+1},w)-\hat{v}(S_t,w).
$$

If we try to treat TD as ``just doing SGD'' on the squared TD error

$$
\ell_t(w) = \frac{1}{2}\,\delta_t^2,
$$

its true gradient is

$$
\nabla_w \ell_t(w)=\delta_t\,\nabla_w \delta_t
=\delta_t\Big(\gamma \nabla_w \hat{v}(S_{t+1},w)-\nabla_w \hat{v}(S_t,w)\Big).
$$

But the standard TD update keeps only the second term:

$$
\Delta w \propto \delta_t\,\nabla_w \hat{v}(S_t,w),
$$

and drops the \(\gamma \nabla_w \hat{v}(S_{t+1},w)\) part. This is called a semi-gradient method: it treats the target \(R_{t+1}+\gamma \hat{v}(S_{t+1},w)\) as if it were a constant, even though it depends on \(w\). This keeps the update local and cheap (it avoids ``backpropagation through the target''), and it matches the Bellman fixed-point view of TD. Because TD is doing a semi-gradient update. We treat the bootstrapped target

$$
R_{t+1}+\gamma \hat{v}(S_{t+1},w)
$$

as a constant at time \(t\) and only differentiate the prediction \(\hat{v}(S_t,w)\).

### Gradient TD

Gradient TD methods address this by defining an explicit objective and then performing (stochastic) gradient descent on it. A common choice is the mean-squared projected Bellman error (MSPBE). Let \(T^\pi\) be the Bellman operator and let \(\Pi\) denote projection onto the function class (e.g., projection onto the span of features under a chosen norm). For an approximate value \(\hat{v}_w\), define

$$
\text{MSPBE}(w)\;=\big\|\Pi T^\pi \hat{v}_w-\hat{v}_w\big\|^2.
$$

Intuitively, \(T^\pi \hat{v}_w\) is the one-step Bellman backup, and \(\Pi\) projects it back into the representable set, so we measure how far \(\hat{v}_w\) is from its projected Bellman update. In the linear case \(\hat{v}_w(s)=x(s)^\top w\), define the TD error

$$
\delta_t = R_{t+1}+\gamma x(S_{t+1})^\top w - x(S_t)^\top w.
$$

This leads to two coupled SGD recursions: one for the main weights \(w\) and one for an auxiliary vector \(h\) that estimates a correction term needed by the gradient. One popular example is GTD2:

$$
h \leftarrow h + \beta\left(\delta_t - x(S_t)^\top h\right)x(S_t),
$$

$$
w \leftarrow w + \alpha\left(x(S_t)-\gamma x(S_{t+1})\right)\big(x(S_t)^\top h\big),
$$

with step sizes \(\alpha,\beta>0\) (often with \(\beta\) larger so \(h\) adapts faster). We can think of it as: standard TD uses one set of weights but may be unstable off-policy; Gradient TD adds \(h\) so the update truly follows the gradient, which improves stability (especially off-policy with linear approximation).

### Convergence of control algorithms (rule of thumb)

When we say an algorithm ``converges'', we usually mean the parameters \(w\) approach a fixed point (and hopefully one that is close to the best possible within the chosen function class). A practical summary for control with function approximation is:

- **Monte Carlo control:** stable with table lookup; often reasonable with linear approximation but may chatter; no general guarantee with non-linear models.
- **SARSA (on-policy TD control):** stable with table lookup; can also chatter with linear approximation; no general guarantee with non-linear models.
- **Q-learning (off-policy TD control):** stable with table lookup; can diverge with linear or non-linear approximation.
- **Gradient Q-learning:** designed for stability with table lookup and linear approximation; still no general guarantee with non-linear models.

Here chattering informally means the parameters keep oscillating around a near-optimal solution rather than settling to an exact fixed point.
