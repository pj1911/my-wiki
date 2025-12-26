# Partially observable Markov decision process (POMDP)

A POMDP is an MDP in which the true state is
hidden. The underlying system still evolves as a Markov chain controlled by actions, but the
agent does not observe \(S_t\) directly. Instead it receives an observation \(O_t\) that provides only
partial information about the state. Equivalently, a POMDP can be viewed as a hidden Markov model
(HMM) with actions.

## Definition

A POMDP is a tuple

$$
\langle \mathcal S,\mathcal A,\mathcal O,\mathcal P,\mathcal R,\mathcal Z,\gamma\rangle,
$$

where

- \(\mathcal S\) is a finite set of (hidden) states,
- \(\mathcal A\) is a finite set of actions,
- \(\mathcal O\) is a finite set of observations,
- \(\mathcal P\) is the controlled transition model:

  
$$
\mathcal P^{a}_{ss'} \;=\; \mathbb P\!\bigl(S_{t+1}=s'\mid S_t=s, A_t=a\bigr),
$$

- \(\mathcal R\) is the reward model (expected one-step reward):

  
$$
\mathcal R^{a}_{s} \;=\; \mathbb E\!\bigl[R_{t+1}\mid S_t=s, A_t=a\bigr],
$$

- \(\mathcal Z\) is the observation (emission) model:

  
$$
\mathcal Z^{a}_{s'o} \;=\; \mathbb P\!\bigl(O_{t+1}=o\mid S_{t+1}=s', A_t=a\bigr),
$$

- \(\gamma\in[0,1]\) is the discount factor.

### What changes relative to an MDP

The dynamics and rewards are still defined on the (hidden) state \(S_t\), but the agent only observes
the stream \(O_t\). Consequently, policies are typically defined on the agent's information (e.g. the
observation--action history, or a belief state), rather than directly on \(S_t\).

## Agent information: histories and belief states

**Histories.**

Because the true state is not directly observed in a POMDP, the agent’s information at time \(t\) is the
entire sequence of past interactions. A history \(H_t\) is the sequence of actions,
observations, and rewards up to time \(t\):

$$
H_t = (A_0, O_1, R_1, \ldots, A_{t-1}, O_t, R_t).
$$

In general, optimal decisions may depend on the full history, not just the most recent observation.

**Belief states.**

Rather than working directly with histories, it is convenient to summarize all relevant information
in a single object. The belief state associated with a history \(h\) is the conditional
distribution of the current (hidden) state given that history:

$$
b(h) \;=\; \bigl(\mathbb P[S_t=s^1\mid H_t=h],\,\ldots,\,\mathbb P[S_t=s^n\mid H_t=h]\bigr).
$$

When \(\mathcal S=\{s^1,\ldots,s^n\}\) is finite, \(b(h)\) is a probability mass function (PMF) over
\(\mathcal S\): it assigns a probability to each discrete state and satisfies \(b_i(h)\ge 0\) and
\(\sum_{i=1}^n b_i(h)=1\). Writing

$$
b(h)=(b_1(h),\ldots,b_n(h)),
\qquad
b_i(h)\;=\;\mathbb P\!\bigl(S_t=s^i \mid H_t=h\bigr),\ i=1,\ldots,n,
$$

the belief represents the agent's uncertainty about the true state after observing \(h\).

**Why belief states matter.**

The belief state is a sufficient statistic for the history: for any action, the distribution of
future states/observations/rewards depends on the past only through \(b(h)\). This lets us treat a POMDP
as a fully observable MDP whose ``state'' is the belief. Consequently, policies can be taken to depend
on \(b(h)\) rather than on the entire history.

## Long-run behavior under a fixed policy: how ergodicity shows up

Once a policy \(\pi\) is fixed, the interaction between the agent and the environment induces a
time-homogeneous stochastic process. In particular:

- in an MDP, the state sequence \((S_t)\) becomes a Markov chain under \(\pi\);
- in a POMDP, although \(S_t\) is hidden, the underlying system still evolves Markovly, and it is
  often useful to analyze the long-run behavior of the induced controlled dynamics (e.g. via the
  belief-MDP viewpoint).

This is why concepts like ergodicity and stationary distributions are natural: they
formalize when the process ``forgets'' its initial condition and admits well-defined long-run averages
(such as average reward).

**Ergodic Markov processes.**

A Markov process is called ergodic if it satisfies:

- Recurrence: every state is visited infinitely often (with probability one).
- Aperiodicity: returns to each state do not occur in a fixed cycle or period.

Intuitively, an ergodic Markov process keeps exploring the entire state space forever and does not get
trapped in cyclic behavior.

**Stationary distribution.**

A probability distribution \(d^\pi\) over \(\mathcal S\) is stationary if it is invariant under
the transition dynamics:

$$
d^\pi(s) \;=\; \sum_{s'\in\mathcal S} d^\pi(s')\,P_{s's},
\qquad
P_{s's}=\mathbb P(S_{t+1}=s \mid S_t=s').
$$

**Fundamental theorem.**

If a Markov process is ergodic, then it admits a unique stationary distribution \(d^\pi\), and the
state distribution converges to it regardless of the initial state:

$$
\mathbb P(S_t=s) \;\longrightarrow\; d^\pi(s)\quad\text{as }t\to\infty.
$$

**Interpretation.**

The stationary distribution \(d^\pi(s)\) describes the long-run fraction of time the process spends in
state \(s\). In reinforcement learning, this distribution is central for defining long-run averages and
understanding the behavior induced by a fixed policy (including, via the belief-MDP view, in POMDPs).

## Ergodic MDPs and the average-reward objective

**Ergodic MDPs.**

An MDP is ergodic if, for every policy \(\pi\), the Markov chain over states induced by
following \(\pi\) is ergodic. Equivalently, under any fixed policy the resulting state process is
recurrent and aperiodic, so it continues to explore the entire state space without getting trapped or
cycling deterministically.

**Average reward.**

In an ergodic MDP, long-run time averages are well defined. For any policy \(\pi\), the average
reward per time step is

$$
\rho^\pi \;=\; \lim_{T\to\infty} \frac{1}{T}\,\mathbb E\!\left[\sum_{t=1}^{T} R_t\right].
$$

A key consequence of ergodicity is that this limit exists and is independent of the initial
state: because the induced Markov chain forgets where it started, the long-run average reward
depends only on the policy \(\pi\).

**Connection to stationary distributions.**

For a fixed policy \(\pi\), let \(d^\pi\) denote the stationary distribution of the induced Markov chain.
Then the average reward can be written as

$$
\rho^\pi \;=\; \sum_{s\in\mathcal S} d^\pi(s)\sum_{a\in\mathcal A}\pi(a\mid s)\,R(s,a).
$$

This makes explicit that \(\rho^\pi\) is the expected reward under the steady-state
behavior of the system, and it is the basis of average-reward reinforcement learning methods. In the
POMDP setting, the same idea is often applied after reducing to the belief-MDP (where the ``state'' is
\(b(h)\)).

## Average-reward value (differential value) and Bellman equation

In the undiscounted setting, a common performance criterion for a policy \(\pi\) is its
average reward per time step

$$
\rho^\pi \;=\; \lim_{T\to\infty}\frac{1}{T}\,\mathbb E_\pi\!\left[\sum_{t=1}^{T} R_t\right],
$$

(when the limit is well defined, e.g. under ergodicity). This scalar captures steady-state
performance, but it does not describe how trajectories from a particular starting state compare to
that baseline.

**Average-reward (differential) value function.**

To quantify start-dependent deviations from the baseline \(\rho^\pi\), we define the
average-reward value function (or differential value function) as

$$
\tilde v_\pi(s)
\;=\;
\mathbb E_\pi\!\left[
\sum_{k=1}^{\infty}\bigl(R_{t+k}-\rho^\pi\bigr)
\;\middle|\;
S_t=s
\right].
$$

It is the expected cumulative excess reward when starting from \(s\), obtained by subtracting
\(\rho^\pi\) at each step and summing the resulting centered rewards.

**Interpretation.**

A positive \(\tilde v_\pi(s)\) indicates that trajectories starting from \(s\) tend to earn above-average
reward (relative to \(\rho^\pi\)) over the transient phase, a negative value indicates below-average
transient behavior. Since only these relative deviations matter, \(\tilde v_\pi\) is defined only
up to an additive constant. One typically fixes a convention such as \(\tilde v_\pi(s_0)=0\) for a
reference state \(s_0\) (or another normalization) to choose a unique representative.

### Average-reward Bellman equation

Using the return recursion and subtracting \(\rho^\pi\), the value function satisfies the
average-reward Bellman equation

$$
\tilde v_\pi(s)=
\mathbb E_\pi\!\left[
(R_{t+1}-\rho^\pi) + \tilde v_\pi(S_{t+1})
\;\middle|\;
S_t=s
\right].
$$

This plays the same role as the Bellman expectation equation in the discounted setting and forms the
basis of average-reward policy evaluation and control.

## Conclusion

A POMDP differs from an MDP only in what the agent observes: the true Markov state \(S_t\) is hidden, so
policies must act on information such as histories \(H_t\) or (more conveniently) beliefs \(b(h)\). The
belief is a sufficient statistic, which lets us reinterpret the POMDP as an MDP on beliefs. Once a
policy is fixed, we obtain an induced Markov process (on states in an MDP, or on beliefs/augmented
state in a POMDP viewpoint). If this induced process is ergodic, it has a unique stationary distribution \(d^\pi\), so long-run
averages such as the average reward \(\rho^\pi\) are well defined. The associated differential value
function \(\tilde v_\pi\) then serves to refine this scalar baseline by quantifying, for each
state \(s\), the expected cumulative deviation from \(\rho^\pi\) (i.e., which states are
transiently better or worse than average). This state-dependent signal is what enables
average-reward policy evaluation and policy improvement, and it is characterized by the
average-reward Bellman equation.
