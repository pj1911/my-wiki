# Introduction to Markov Decision Processes

In reinforcement learning (RL), interaction with an environment is modeled as a random trajectory

$$  
S_0, A_0, R_1, S_1, A_1, R_2, S_2, \dots  
$$

where $S_t \in \mathcal{S}$ is the state at time $t$, $A_t \in \mathcal{A}$ is the action chosen at time $t$, and $R_{t+1} \in \mathbb{R}$ is the reward received after taking $A_t$ in $S_t$ and moving to $S_{t+1}$.

A Markov Decision Process (MDP) is a standard mathematical model for the environment in RL. An MDP assumes the environment is fully observable, meaning the agent can observe the true state. Equivalently, the state is intended to summarize all information needed to predict what can happen next, so the past matters only through the current state. Because of this, many RL problems can be formulated as MDPs.

A key point to note here is that the trajectory variables $S_0, A_0, R_1, S_1, A_1, R_2, \dots$ are random variables, so there is a joint distribution over the entire sequence. Equivalently, for any particular full trajectory $
(S_0, A_0, R_1, S_1, A_1, R_2, \dots),$
the model assigns a well-defined probability to observing exactly that sequence.

$$  
\Pr(S_{t+1}=s' \mid S_0,A_0,R_1,\dots,S_t=s, A_t=a).  
$$

Rather than writing the full joint distribution over an entire trajectory explicitly as above, we usually describe the process through one-step conditional distribution.

$$  
\Pr(S_{t+1}=s' \mid S_0,A_0,R_1,\dots,S_t=s, A_t=a)
= \Pr(S_{t+1}=s' \mid S_t=s, A_t=a).  
$$

This is the Markov property. The key idea here is that the future depends only on the present, not on the full past, assuming a state captures all information from the history that is relevant for predicting the future. Often we also include reward in the Markov property:

$$  
\Pr(S_{t+1}=s', R_{t+1}=r \mid \text{history up to } t,\, S_t=s, A_t=a) =
\Pr(S_{t+1}=s', R_{t+1}=r \mid S_t=s, A_t=a).  
$$

This means the distribution of the next state depends only on the current state. Equivalently, the state is a sufficient statistic of the future.

**Fully observable environment assumption.**
When the environment is fully observable and we have chosen a state variable $S_t$ that captures everything relevant, the Markov property is a reasonable modeling assumption. If our $S_t$ omits important information, the process may fail to be Markov (this motivates POMDPs and state augmentation, which will be discussed in a later chapter).

### State Transition Matrix

In a Markov process, state changes are described using transition probabilities. Assume a finite state space \(\mathcal{S}=\{1,2,\dots,n\}\) and (time-homogeneous) transitions. For a current state \(s\) and a next state \(s'\), the state transition probability is given as

$$  
P_{ss'} \;=\; \mathbb{P}(S_{t+1}=s' \mid S_t=s),
\qquad s,s' \in \{1,\dots,n\}.  
$$

Collecting all transition probabilities gives the state transition matrix \(P \in \mathbb{R}^{n\times n}\).

$$  
P =
\begin{bmatrix}
P_{11} & P_{12} & \cdots & P_{1n} \\
P_{21} & P_{22} & \cdots & P_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
P_{n1} & P_{n2} & \cdots & P_{nn}
\end{bmatrix}.  
$$

Here, row \(s\) is the conditional distribution over the next state given the current state \(s\), so its entries satisfy \(P_{ss'} \ge 0\) and

$$  
\sum_{s'=1}^n P_{ss'} = 1.  
$$

Meanwhile, column \(s'\) collects the probabilities of transitioning into \(s'\) from each possible current state,

$$  
(P_{1s'}, P_{2s'}, \dots, P_{ns'})^\top,  
$$

so columns are not required to sum to \(1\) in general. Instead, they indicate how likely it is to arrive at \(s'\) from each origin, and in stationary analysis they help interpret how probability mass moves under repeated application of \(P\).

#### Time dependent trasnsition matrix

Note that the above formulation assumes that the transition probabilities are constant over time and is therefore called a time-homogeneous Markov Process. If the transition law depends on time, we have a time-inhomogeneous Markov process given by:

$$  
\Pr(S_{t+1}=s' \mid S_t=s) = P_t(s' \mid s),  
$$

or in matrix form \(P_t\) instead of a single \(P\). It is usually handeled in the following two ways:

**(1) Accept nonstationarity.**
We explicitly model \(P_t\). This is common in nonstationary environments. Many RL algorithms assume stationarity for guarantees. If the environment changes, performance guarantees weaken, but the process can still be treated as Markov-with-time.

**(2) Augment the state to recover stationarity.**
If the change is systematic and depends on something observable (like time of day, season, or a known mode), we can include that in the state and define an augmented state as:

$$  
\tilde{S}_t = (S_t, t).  
$$

Then the process can become time-homogeneous in the augmented space:

$$  
\Pr(\tilde{S}_{t+1} \mid \tilde{S}_t) \;\text{does not need explicit dependence on } t \text{ anymore}  
$$

because \(t\) is now part of the state. More generally, if the environment has a hidden ``mode'' \(M_t\) that drives changes, we can try to infer and include (an estimate of) \(M_t\) in the state.

**Key idea.**
Nonstationarity often signals that our current \(S_t\) is not capturing all relevant variables. ``Fixing'' it is usually done by adding missing variables to the state representation.

## Markov Reward Process

A Markov chain models how states evolve. To reason about good and bad states, we add rewards. A Markov Reward Process (MRP) extends a Markov process by attaching a numerical reward and is defined by the tuple \((\mathcal{S}, P, R, \gamma)\), where:

- \(\mathcal{S}\) is a finite set of states (the state space).
- \(P\) is the state transition model (matrix form), with entries

$$  
P_{ss'} = \mathbb{P}(S_{t+1} = s' \mid S_t = s).  
$$

- \(R\) is the reward model. A common convention is state-based reward,

$$  
R_s = \mathbb{E}[R_{t+1} \mid S_t = s],  
$$

which gives the expected immediate reward when in state \(s\). Another common convention is transition-based reward,

$$  
R(s,s') = \mathbb{E}[R_{t+1}\mid S_t=s,\, S_{t+1}=s'],  
$$

which allows the reward to depend on the transition.
- \(\gamma \in [0,1]\) is the discount factor, which controls how much future rewards are valued relative to immediate rewards.

An MRP can be seen as a Markov chain with rewards attached to states (or, under the transition-based convention, attached to transitions).

**Remark (model-based vs. model-free).**
In the definition of an MRP, the tuple \((\mathcal S,P,R,\gamma)\) specifies the environment, so \(P\) and the reward model \(R\) are treated as given. In many RL settings, however, \(P\) and the expected reward model (e.g. \(R_s=\mathbb{E}[R_{t+1}\mid S_t=s]\)) are unknown. We only observe sampled transitions and realized rewards \((S_t,R_{t+1},S_{t+1})\). The discount factor \(\gamma\) is typically chosen by the practitioner.

### Return: total reward over time

In reinforcement learning we often consider an episode (also called a trajectory), which is a single sampled sequence of states, actions, and rewards generated by interacting with the environment:

$$  
S_0, A_0, R_1, S_1, A_1, R_2, \dots  
$$

The episode may be finite (ending at a terminal time \(T\)) or infinite. To compute the return at time \(t\), we take the discounted sum of rewards received from time \(t+1\) onward:

$$  
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+1+k}.  
$$

Special cases and common settings:

- Undiscounted return (\(\gamma=1\)):

$$  
G_t = \sum_{k=0}^{\infty} R_{t+1+k}.  
$$

This is used when we want to value rewards at all future times equally (no time preference). In practice it is most common in episodic tasks with terminal states, where the episode ends so the sum is over finitely many rewards and stays well-defined.
- Finite-horizon return (horizon \(T\)):

$$  
G_t^{(T)} = \sum_{k=0}^{T-t-1} \gamma^k R_{t+1+k}.  
$$

This is used when only rewards up to a deadline matter. For example, in problems with a fixed-length episode, limited budget of steps, or a task where performance is evaluated over the next \(T-t\) time steps, rewards after time \(T\) are not counted at all.

#### Why discounting?

Discounting is often motivated by three (compatible) reasons:

**1) Keep returns bounded in cyclic processes.**
In a Markov process with cycles, we can revisit rewarding states infinitely often.
If rewards are nonnegative and we use \(\gamma=1\), the infinite sum can diverge.

Example: if \(R_{t+1}=1\) forever, then

$$  
\sum_{k=0}^{\infty} 1 = \infty,  
$$

but with \(\gamma \in (0,1)\),

$$  
\sum_{k=0}^{\infty} \gamma^k = \frac{1}{1-\gamma} < \infty.  
$$

So discounting makes infinite-horizon problems mathematically well-behaved.

**2) Present value (finance analogy).**
A reward received later is worth less today. Discounting models this by down-weighting future outcomes.

**3) Immediate vs delayed under uncertainty / imperfect models.**
When the world is uncertain (and our model isn't perfect), far-future predictions are less reliable.
Discounting implicitly says: optimize what we can predict/control more confidently.

### Value function: expected return from a state

The state-value function (or simply value function) of an MRP is the expected discounted return when starting in state \(s\):

$$  
V(s) = \mathbb{E}\!\left[G_t \mid S_t=s\right],
\qquad
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+1+k}.  
$$

Equivalently, \(V(s)\) is a function \(V:\mathcal S\to\mathbb R\) that assigns to each state \(s\) the long-term utility of being in \(s\) under the MRP dynamics (no actions), with future rewards geometrically discounted by \(\gamma\). This expectation is over the randomness of future transitions (and rewards).

#### Bellman expectation equation (MRP)

We can split the return into the next reward plus the rest:

$$  
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+1+k}
= R_{t+1} + \sum_{k=1}^{\infty} \gamma^k R_{t+1+k}
= R_{t+1} + \gamma \sum_{j=0}^{\infty} \gamma^j R_{t+2+j}
= R_{t+1} + \gamma G_{t+1}.  
$$

Take conditional expectation given \(S_t=s\):

$$  
V(s) = \mathbb{E}\left[G_t \mid S_t=s\right]= \mathbb{E}[R_{t+1}\mid S_t=s] + \gamma\,\mathbb{E}[G_{t+1}\mid S_t=s].  
$$

Now use the law of total expectation (proof shown in appendix) over the next state \(S_{t+1}\):

$$  
\mathbb{E}[G_{t+1}\mid S_t=s]=
\sum_{s'} \Pr(S_{t+1}=s'\mid S_t=s)\,\mathbb{E}[G_{t+1}\mid S_{t+1}=s',\,S_t=s].  
$$

Since under the Markov property (and a fixed stationary policy) the future return \(G_{t+1}\) depends on the past only through the current state \(S_{t+1}\), we have

$$  
\mathbb{E}[G_{t+1}\mid S_{t+1}=s',\,S_t=s]
=\mathbb{E}[G_{t+1}\mid S_{t+1}=s']
=V(s').  
$$

Then we can write:

$$  
\mathbb{E}[G_{t+1}\mid S_t=s]
=\sum_{s'} \Pr(S_{t+1}=s'\mid S_t=s)\,V(s').  
$$

Substituting this back into

$$  
V(s)=\mathbb{E}[R_{t+1}\mid S_t=s]+\gamma\,\mathbb{E}[G_{t+1}\mid S_t=s],  
$$

gives

$$  
V(s)=\mathbb{E}[R_{t+1}\mid S_t=s]+\gamma\sum_{s'} \Pr(S_{t+1}=s'\mid S_t=s)\,V(s').  
$$

Identifying

$$  
\mathcal{R}_s := \mathbb{E}[R_{t+1}\mid S_t=s]
\quad \text{and} \quad
\mathcal{P}_{ss'} := \Pr(S_{t+1}=s' \mid S_t=s),  
$$

this becomes

$$  
v(s) = \mathcal{R}_s + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}\, v(s'),  
$$

which is the Bellman expectation equation.

In vector form (finite state space \(\mathcal S\) with \(|\mathcal S|=n\)), let \(v\in\mathbb{R}^n\) with entries \(v_s=v(s)\), let \(\mathcal R\in\mathbb{R}^n\) with entries \(\mathcal R_s\), and let \(\mathcal P\in\mathbb{R}^{n\times n}\) with entries \(\mathcal P_{ss'}\). Then

$$  
v = \mathcal R + \gamma \mathcal P\,v
\quad\Longrightarrow\quad
(I-\gamma \mathcal P)\,v=\mathcal R
\quad\Longrightarrow\quad
v=(I-\gamma \mathcal P)^{-1}\mathcal R,  
$$

where \(I\) is the \(n\times n\) identity matrix. For \(\gamma\in[0,1)\) and finite MRPs, \((I-\gamma\mathcal P)\) is invertible.

#### Bellman equation for MRPs (recursive form)

The Bellman equation expresses the value function as a recursion: the value of a state \(s\) is written in terms of the values of its possible successor states. Starting from \(s\), the process transitions to a random next state \(s'\), yields an immediate reward, and then continues recursively from \(s'\). Taking expectations over all possible next states yields a recursive equation for \(V(s)\).

\medskip
\noindent Recursive form (transition-based reward, most general).
Starting from the definition

$$  
V(s)=\mathbb{E}[G_t\mid S_t=s],\qquad 
G_t=R_{t+1}+\gamma G_{t+1},  
$$

take conditional expectation given \(S_t=s\):

$$  
V(s)=\mathbb{E}[R_{t+1}\mid S_t=s]+\gamma\,\mathbb{E}[G_{t+1}\mid S_t=s].  
$$

Apply the law of total expectation over \(S_{t+1}\):

$$  
\mathbb{E}[R_{t+1}\mid S_t=s]
=\sum_{s'}P(s'\mid s)\,\mathbb{E}[R_{t+1}\mid S_t=s,S_{t+1}=s']
=\sum_{s'}P(s'\mid s)\,R(s,s'),  
$$

and (Markov property)

$$  
\mathbb{E}[G_{t+1}\mid S_t=s]
=\sum_{s'}P(s'\mid s)\,\mathbb{E}[G_{t+1}\mid S_{t+1}=s',S_t=s]
=\sum_{s'}P(s'\mid s)\,V(s').  
$$

Substituting both back gives

$$  
V(s)=\sum_{s'}P(s'\mid s)\Big(R(s,s')+\gamma V(s')\Big).  
$$

This equation shows that \(V(s)\) is obtained as a probability-weighted average of the quantities
\(R(s,s')+\gamma V(s')\) associated with each possible successor state \(s'\).

\medskip
\noindent Special case (state-based reward).
If rewards depend only on the current state, i.e. \(R(s,s')\equiv R(s)\), the equation reduces to

$$  
V(s)=R(s)+\gamma\sum_{s'\in\mathcal S}P(s'\mid s)\,V(s').  
$$

\medskip
\noindent Worked example (recursive value computation).
Suppose from state \(s\) the next states are \(\{a,b,c\}\) with

$$  
P(a\mid s)=0.2,\qquad P(b\mid s)=0.5,\qquad P(c\mid s)=0.3.  
$$

Using the transition-based Bellman equation,

$$  
V(s)
=0.2\big(R(s,a)+\gamma V(a)\big)
+0.5\big(R(s,b)+\gamma V(b)\big)
+0.3\big(R(s,c)+\gamma V(c)\big).  
$$

Equivalently, this means compute the quantity \(R(s,s')+\gamma V(s')\) for each successor \(s'\in\{a,b,c\}\) and take their probability-weighted average.

#### Complexity of solving the Bellman equation

For a finite MRP with \(n=|\mathcal S|\) states, the Bellman equation in vector form is

$$  
v=\mathcal R+\gamma \mathcal P v
\quad\Longleftrightarrow\quad
(I-\gamma \mathcal P)v=\mathcal R.  
$$

**Direct (matrix) solution.**
If we treat \(\mathcal P\) as a dense \(n\times n\) matrix, solving the linear system \((I-\gamma \mathcal P)v=\mathcal R\) with standard dense methods typically costs \(\mathcal O(n^3)\) time and storing \(\mathcal P\) costs \(\mathcal O(n^2)\) memory. Thus direct solvers are practical mainly for small MRPs (or when \(\mathcal P\) is very sparse and we can use sparse linear algebra).

**Large MRPs: iterative and sample-based methods.**
When \(n\) is large, we usually avoid forming and inverting matrices explicitly. Instead we use iterative methods that repeatedly apply the Bellman recursion (e.g. value iteration / iterative policy evaluation), or model-free methods that estimate \(v\) from sampled trajectories (e.g. Monte Carlo and temporal-difference learning) without needing an explicit \(\mathcal P\).

#### Evaluation methods for large MRPs

When the state space is large, directly solving the Bellman equation is often impractical. Instead, value functions are computed approximately using iterative or sample-based methods. The three most common approaches for evaluating the value function of an MRP differ in what is assumed to be known and how the Bellman recursion is used (these will be discussed in detail in later chapters).

**1) Dynamic Programming (DP) / iterative evaluation.**
Assume the MRP model \((\mathcal P,\mathcal R)\) is known. Starting from an arbitrary initial estimate \(v^{(0)}\), repeatedly apply the Bellman recursion

$$  
v^{(k+1)} = \mathcal R + \gamma \mathcal P\, v^{(k)}.  
$$

Component-wise,

$$  
v^{(k+1)}(s)=\mathcal R_s+\gamma\sum_{s'}\mathcal P_{ss'}\,v^{(k)}(s').  
$$

Under standard conditions, \(v^{(k)}\) converges to the true value function \(v\).

**2) Monte Carlo (MC) evaluation.**
Assume the model \((\mathcal P,\mathcal R)\) is unknown, but sample episodes can be generated. Since
\(
v(s)=\mathbb E[G_t\mid S_t=s],
\)
estimate \(v(s)\) by averaging observed returns. If \(m\) visits to state \(s\) are observed, the Monte Carlo estimate is

$$  
\hat v(s)=\frac{1}{m}\sum_{i=1}^{m} G^{(i)},
\qquad
G^{(i)}=\sum_{k=0}^{\infty}\gamma^k R^{(i)}_{t_i+1+k}.  
$$

Monte Carlo methods are purely sample-based and do not require explicit knowledge of \(\mathcal P\) or \(\mathcal R\).

**3) Temporal-Difference (TD) learning.**
Temporal-Difference methods combine ideas from DP (bootstrapping) and MC (sampling). From a single observed transition
\(
S_t=s \to S_{t+1}=s'
\)
with reward \(R_{t+1}\), define the TD error

$$  
\delta_t = R_{t+1} + \gamma v(s') - v(s).  
$$

The value estimate is updated incrementally using a step size \(\alpha\in(0,1]\):

$$  
v(s)\leftarrow v(s)+\alpha\,\delta_t.  
$$

TD methods update values online, do not require full episodes, and do not assume knowledge of the transition model.

## Markov Decision Processes (MDPs)

So far we considered Markov Reward Processes (MRPs), where the dynamics are fixed and there is no control: the process moves according to \(\mathcal P\) and generates rewards according to \(\mathcal R\) (or \(R(s,s')\)). A Markov Decision Process (MDP) generalizes an MRP by introducing actions, so that the transition and reward distributions can depend on the agent's choice. A (finite, discounted) MDP is a tuple

$$  
(\mathcal S,\mathcal A, P, R, \gamma),  
$$

where \(\mathcal S\) is the finite state space, \(\mathcal A\) is a finite action set, \(\gamma\in[0,1)\) is the discount factor, and:

$$  
P(s'\mid s,a) = \Pr(S_{t+1}=s'\mid S_t=s, A_t=a)  
$$

is the action-dependent transition model. In the finite case, it is convenient to view \(P(\cdot\mid\cdot,a)\) as a matrix for each action:

$$  
\mathcal P^{a}_{ss'} = P(s'\mid s,a),
\qquad a\in\mathcal A,  
$$

i.e., an MDP has a collection of transition matrices \(\{\mathcal P^{a}\}_{a\in\mathcal A}\) rather than a single \(\mathcal P\) as in an MRP. Rewards may also depend on the chosen action. Common conventions are

$$  
\mathcal R(s,a)= \mathbb E[R_{t+1}\mid S_t=s, A_t=a],
\qquad\text{or}\qquad
\mathcal R(s,a,s')= \mathbb E[R_{t+1}\mid S_t=s, A_t=a, S_{t+1}=s'],  
$$

Thus, compared to an MRP, the new ingredient in an MDP is that both the transition dynamics and the reward distribution can change with the action \(a\).

### Policies: how the agent behaves

In an MDP, the environment specifies what can happen given a state and an action via \(P(s'\mid s,a)\) and how rewards are generated via \(R(\cdot)\). What remains is to specify the agent's behavior: how actions are selected in each state. This is captured by a policy.

**Definition (deterministic policy).**
A deterministic policy chooses a single action in each state:

$$  
\mu:\mathcal S\to\mathcal A,\qquad A_t=\mu(S_t).  
$$

**Definition (stochastic policy).**
A stochastic policy is a mapping from states to distributions over actions:

$$  
\pi(a\mid s)=\Pr(A_t=a\mid S_t=s),
\qquad s\in\mathcal S,\ a\in\mathcal A.  
$$

Thus, for each state \(s\), \(\pi(\cdot\mid s)\) is a probability distribution on \(\mathcal A\).

**Stationary vs. non-stationary.**
A policy is stationary if it does not explicitly depend on time:

$$  
\pi_t(a\mid s)=\pi(a\mid s)\quad \forall t,  
$$

and non-stationary if it can vary with time:

$$  
\pi_t(a\mid s)\neq \pi_{t'}(a\mid s)\ \text{for some }t\neq t'.  
$$

Non-stationary policies are most common in finite-horizon settings, while discounted infinite-horizon problems typically focus on stationary policies.

#### Induced Markov processes under a policy

An MDP specifies how the environment responds to state--action pairs. Once a policy is fixed, the agent's behavior is fully determined, and the remaining randomness comes only from the environment. In this sense, a policy ``removes control'' from the MDP and induces a Markov process. In this section we formalize this idea and show that any fixed policy transforms an MDP into a Markov reward process (MRP).

**State dynamics under a fixed policy.**
Fix an MDP \((\mathcal S,\mathcal A,\mathcal P,\mathcal R,\gamma)\) and a (stationary) policy \(\pi\). The resulting state sequence \(\{S_t\}\) is a Markov chain with transition probabilities

$$  
\mathcal P^\pi_{ss'}
\;=
\Pr(S_{t+1}=s'\mid S_t=s).  
$$

By marginalizing over the action chosen by the policy,

$$  
\mathcal P^\pi_{ss'}
= \sum_{a\in\mathcal A}
\Pr(S_{t+1}=s'\mid S_t=s,A_t=a)\Pr(A_t=a\mid S_t=s)
= \sum_{a\in\mathcal A}\pi(a\mid s)\,\mathcal P^{a}_{ss'}.  
$$

Thus, a fixed policy induces a single state-transition matrix \(\mathcal P^\pi\).

**Rewards under a fixed policy.**
Similarly, the expected one-step reward under policy \(\pi\) is

$$  
\mathcal R^\pi_s
\;=\;
\mathbb E[R_{t+1}\mid S_t=s]
= \sum_{a\in\mathcal A}\pi(a\mid s)\,\mathcal R(s,a),  
$$

or, under a transition-based reward convention,

$$  
\mathcal R^\pi_s
= \sum_{a\in\mathcal A}\pi(a\mid s)\sum_{s'\in\mathcal S}\mathcal P^{a}_{ss'}\,\mathcal R(s,a,s').  
$$

Combining the induced transition matrix \(\mathcal P^\pi\) and reward function \(\mathcal R^\pi\), a fixed policy \(\pi\) turns the MDP into a Markov reward process

$$  
(\mathcal S,\mathcal P^\pi,\mathcal R^\pi,\gamma).  
$$

This induced MRP allows us to evaluate the policy using the value-function machinery developed earlier.

### Value functions in MDPs: state-value and action-value

We keep the same notion of discounted return as in MRPs:

$$  
G_t = \sum_{k=0}^{\infty}\gamma^k R_{t+1+k}.  
$$

Given a fixed policy \(\pi\), we evaluate behavior using two closely related value functions.

**State-value function.**
The state-value function under policy \(\pi\) is the expected return when starting in state \(s\) and then following \(\pi\):

$$  
V^\pi(s)= \mathbb{E}_\pi\!\left[G_t \mid S_t=s\right].  
$$

**Action-value function.**
The action-value function under policy \(\pi\) is the expected return when starting in state \(s\), taking action \(a\) at time \(t\), and then following \(\pi\) thereafter:

$$  
Q^\pi(s,a)= \mathbb{E}_\pi\!\left[G_t \mid S_t=s,\,A_t=a\right].  
$$

### Bellman expectation equations in MDPs

The derivations mirror the MRP case: the return satisfies the recursion \(G_t=R_{t+1}+\gamma G_{t+1}\), and we take conditional expectations. The difference is that in an MDP the next state and reward depend on the action, and actions are chosen according to the policy \(\pi\).

**Bellman expectation equation for \(V^\pi\).**
Start from

$$  
V^\pi(s)= \mathbb E_\pi[G_t\mid S_t=s],\qquad G_t=R_{t+1}+\gamma G_{t+1}.  
$$

Taking \(\mathbb E_\pi[\cdot\mid S_t=s]\) gives

$$  
V^\pi(s)=\mathbb E_\pi[R_{t+1}\mid S_t=s]+\gamma\,\mathbb E_\pi[G_{t+1}\mid S_t=s].  
$$

Condition on the first action \(A_t\) and use \(\Pr(A_t=a\mid S_t=s)=\pi(a\mid s)\):

$$  
\mathbb E_\pi[R_{t+1}\mid S_t=s]
=\sum_{a\in\mathcal A}\pi(a\mid s)\,\mathbb E[R_{t+1}\mid S_t=s,A_t=a]
=\sum_{a}\pi(a\mid s)\,R(s,a).  
$$

Similarly, expand \(\mathbb E_\pi[G_{t+1}\mid S_t=s]\) by conditioning on the first action and next state. First apply the law of total expectation over \(A_t\) (proof given in appendix):

$$  
\mathbb E_\pi[G_{t+1}\mid S_t=s]
=\sum_{a\in\mathcal A}\Pr_\pi(A_t=a\mid S_t=s)\,\mathbb E_\pi[G_{t+1}\mid S_t=s,A_t=a]
=\sum_{a\in\mathcal A}\pi(a\mid s)\,\mathbb E_\pi[G_{t+1}\mid S_t=s,A_t=a].  
$$

Next, for each fixed \((s,a)\), apply the law of total expectation over \(S_{t+1}\):

$$  
\mathbb E_\pi[G_{t+1}\mid S_t=s,A_t=a]
=\sum_{s'\in\mathcal S}\Pr(S_{t+1}=s'\mid S_t=s,A_t=a)\,
\mathbb E_\pi[G_{t+1}\mid S_t=s,A_t=a,S_{t+1}=s'].  
$$

Using the Markov property, once we condition on \(S_{t+1}=s'\), the future return from time \(t+1\) onward does not depend on \((S_t,A_t)\), so

$$  
\mathbb E_\pi[G_{t+1}\mid S_t=s,A_t=a,S_{t+1}=s']
=\mathbb E_\pi[G_{t+1}\mid S_{t+1}=s']
=V^\pi(s').  
$$

Therefore,

$$  
\mathbb E_\pi[G_{t+1}\mid S_t=s,A_t=a]
=\sum_{s'\in\mathcal S}P(s'\mid s,a)\,V^\pi(s'),  
$$

and substituting back yields

$$  
\mathbb E_\pi[G_{t+1}\mid S_t=s]
=\sum_{a\in\mathcal A}\pi(a\mid s)\sum_{s'\in\mathcal S}P(s'\mid s,a)\,V^\pi(s').  
$$

Substituting these two expressions yields the Bellman expectation equation:

$$  
V^\pi(s)
= \sum_{a\in\mathcal A}\pi(a\mid s)\left[\,R(s,a)+\gamma\sum_{s'\in\mathcal S}P(s'\mid s,a)\,V^\pi(s')\right].  
$$

**Bellman expectation equation for \(Q^\pi\).**
Define

$$  
Q^\pi(s,a)= \mathbb E_\pi[G_t\mid S_t=s,A_t=a].  
$$

Using \(G_t=R_{t+1}+\gamma G_{t+1}\) and conditioning on \(S_{t+1}\),

$$  
Q^\pi(s,a)
=\mathbb E[R_{t+1}\mid S_t=s,A_t=a]+\gamma\,\mathbb E_\pi[G_{t+1}\mid S_t=s,A_t=a]
=R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,V^\pi(s').  
$$

Using \(V^\pi(s')=\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')\), this can be written as a recursion in \(Q^\pi\) alone:

$$  
Q^\pi(s,a)=R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\sum_{a'}\pi(a'\mid s')\,Q^\pi(s',a').  
$$

**Relationship between \(V^\pi\) and \(Q^\pi\).**
Probability-weighted sum of \(Q^\pi(s,a)\) over the action drawn from \(\pi(\cdot\mid s)\) recovers \(V^\pi(s)\):

$$  
V^\pi(s)=\sum_{a\in\mathcal A}\pi(a\mid s)\,Q^\pi(s,a).  
$$

These equations reduce to the original MRP Bellman recursion when there is only one available action (no control).

#### Matrix form of \(V^\pi\) and \(Q^\pi\) (finite MDP)

Assume \(|\mathcal S|=n\) and \(|\mathcal A|=m\). Stack the state-values into a vector \(v^\pi\in\mathbb R^n\) with entries \(v^\pi_s=V^\pi(s)\).

**Induced MRP (state-value).**
Define the policy-induced transition matrix \(\mathcal P^\pi\in\mathbb R^{n\times n}\) and reward vector \(\mathcal R^\pi\in\mathbb R^n\) by

$$  
\mathcal P^\pi_{ss'}= \sum_{a\in\mathcal A}\pi(a\mid s)\,\mathcal P^{a}_{ss'},
\qquad
\mathcal R^\pi_s= \sum_{a\in\mathcal A}\pi(a\mid s)\,\mathcal R(s,a),  
$$

(analogously if rewards depend on \((s,a,s')\)). Then the Bellman expectation equation is

$$  
v^\pi=\mathcal R^\pi+\gamma \mathcal P^\pi v^\pi
\quad\Longrightarrow\quad
v^\pi=(I-\gamma \mathcal P^\pi)^{-1}\mathcal R^\pi,  
$$

when \((I-\gamma \mathcal P^\pi)\) is invertible (e.g. for \(\gamma\in[0,1)\) in the finite case).

**Action-value (state--action form).**
Stack action-values into a vector \(q^\pi\in\mathbb R^{nm}\) indexed by \((s,a)\) with entries \(q^\pi_{(s,a)}=Q^\pi(s,a)\). Define

$$  
r_{(s,a)}= \mathcal R(s,a),
\qquad
\mathcal P_{(s,a),s'}= P(s'\mid s,a),  
$$

and let \(\Pi\in\mathbb R^{n\times (nm)}\) be the policy-averaging matrix with entries

$$  
\Pi_{s,(s,a)}= \pi(a\mid s).  
$$

Then \(v^\pi=\Pi q^\pi\) and the Bellman equation
\(Q^\pi(s,a)=\mathcal R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^\pi(s')\)
becomes

$$  
q^\pi = r + \gamma\,\mathcal P\,v^\pi
      = r + \gamma\,\mathcal P\,\Pi\,q^\pi
\quad\Longrightarrow\quad
q^\pi=(I-\gamma \mathcal P\Pi)^{-1}r,  
$$

when the inverse exists.

### Policy evaluation

In the previous section we derived the Bellman expectation equations for an MDP under a fixed policy
\(\pi\), and we ended by writing them in matrix form. At this point, it helps to pause and make explicit
what we are actually doing: we are no longer making decisions. The policy \(\pi\) has already committed
to how actions are chosen in each state, so the only thing left is to measure how good that
committed behavior is. This is called Policy evaluation:

$$  
\text{given }\pi,\ \text{compute }V^\pi\text{ and }Q^\pi.  
$$

In other words, we are simply asking what long-run discounted return this particular policy achieves.
In the finite discounted setting, the matrix form is not just a compact way to write the Bellman
equations: it also tells us something reassuring. There is exactly one function \(V^\pi\) that
satisfies them, so policy evaluation is not ambiguous.

**Existence and uniqueness of \(V^\pi\) (finite discounted case)**
\label{prop:eval-unique}
Assume \(|\mathcal S|<\infty\), \(|\mathcal A|<\infty\), and \(\gamma\in[0,1)\). For any fixed policy \(\pi\),
the Bellman expectation equations admit a unique solution \(V:\mathcal S\to\mathbb R\). This
solution is exactly the value function \(V^\pi\). The same statement can be made for \(Q^\pi\). (Proof is beyond our scope)

So far, we have been answering: "If I behave according to \(\pi\), what do I get?" The next
step is the real control question: "Can we do better?" That means comparing policies and
pushing performance as high as possible. This is where optimal value functions and an
optimal policy enter, and where the Bellman equations change from averaging over actions to
taking a maximum.

### Optimality in MDPs: from evaluation to control

So far we have focused on policy evaluation: given a fixed policy \(\pi\), we quantify its long-term performance via

$$
V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s],
\qquad
Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].
$$

The control problem asks the complementary question: among all policies, which behavior is best?

#### Optimal value functions and optimal policies

Let \(\Pi\) denote the set of all (possibly stochastic) policies. Define the optimal state-value function and optimal action-value function by

$$
\begin{aligned}
V^*(s)&= \sup_{\pi\in\Pi} V^\pi(s),\\
Q^*(s,a)&= \sup_{\pi\in\Pi} Q^\pi(s,a).
\end{aligned}
$$

Equivalently,

$$
V^*(s)=\sup_{\pi\in\Pi}\mathbb{E}_\pi\!\left[\sum_{k=0}^{\infty}\gamma^k R_{t+1+k}\,\middle|\,S_t=s\right],
$$

and

$$
Q^*(s,a)=\sup_{\pi\in\Pi}\mathbb{E}_\pi\!\left[\sum_{k=0}^{\infty}\gamma^k R_{t+1+k}\,\middle|\,S_t=s,A_t=a\right].
$$

Intuitively, \(Q^*(s,a)\) is the best achievable return if we force the first action to be \(a\) in state \(s\)
and then behave optimally thereafter.

**Optimal policy.**
A policy \(\pi^*\) is optimal if it achieves the optimal value from every state:

$$
V^{\pi^*}(s)=V^*(s)\quad\text{for all }s\in\mathcal S.
$$

Equivalently,

$$
Q^{\pi^*}(s,a)=Q^*(s,a)\quad\text{for all }(s,a)\in\mathcal S\times\mathcal A.
$$

**Existence (finite discounted case).**
In finite discounted MDPs (\(|\mathcal S|<\infty\), \(|\mathcal A|<\infty\), \(\gamma\in[0,1)\)), there exists
at least one optimal policy. Moreover, there always exists an optimal deterministic stationary policy.
Optimal policies need not be unique: it is common to have multiple distinct policies that all achieve \(V^*\).
(Proof is beyond our scope)

In particular, in this setting the suprema above are attained, so we may write

$$
V^*(s)=\max_{\pi\in\Pi}V^\pi(s),\qquad Q^*(s,a)=\max_{\pi\in\Pi}Q^\pi(s,a).
$$

**Why \(Q^*\) is especially useful.**
Since \(Q^*(s,a)\) already accounts for optimal future behavior, choosing an action that maximizes
\(Q^*(s,a)\) is optimal for the current state:

$$
a^*(s)\in\arg\max_{a\in\mathcal A} Q^*(s,a),
\qquad
\pi^*(a\mid s)=
\begin{cases}
1, & a\in\arg\max_{a'} Q^*(s,a'),\\
0, & \text{otherwise}.
\end{cases}
$$

#### Bellman optimality equations

The derivation mirrors the Bellman expectation equations. Start from the return recursion

$$
G_t = R_{t+1}+\gamma G_{t+1},
$$

take conditional expectations, and apply the law of total expectation over the next state. The only
conceptual change is optimality: instead of averaging over actions using \(\pi(\cdot\mid s)\), we
select the action that maximizes the expected continuation value.

**Optimal state-value recursion.**
If we take action \(a\) now in state \(s\) and then behave optimally thereafter, the expected return is

$$
R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,V^*(s').
$$

Choosing the best action gives

$$
V^*(s)=\max_{a\in\mathcal A}\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,V^*(s')\right].
$$

**Optimal action-value recursion.**
If the first action is fixed to be \(a\) in state \(s\), then after transitioning to \(s'\) optimal behavior
chooses the best next action:

$$
Q^*(s,a)=R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,\max_{a'}Q^*(s',a').
$$

**Consistency and greedy optimality.**
The link between \(V^*\) and \(Q^*\) is immediate:

$$
V^*(s)=\max_{a\in\mathcal A}Q^*(s,a),
\qquad
\pi^*(s)\in\arg\max_{a\in\mathcal A}Q^*(s,a).
$$

and plugging this into the \(Q^*\) equation gives the full recursion:

$$
Q^*(s,a) =
R(s,a)
+
\gamma \sum_{s'} P(s'\mid s,a)\,V^*(s')
\quad\text{with}\quad
V^*(s')=\max_{a'}Q^*(s',a').
$$

Indeed, by definition \(Q^*(s,a)\) is the optimal return conditional on taking \(a\) first, so the
best achievable value from \(s\) is obtained by selecting the best first action, which gives
\(V^*(s)=\max_a Q^*(s,a)\). Conversely, any policy that always picks an action attaining this maximum in
each state is optimal, because \(Q^*\) already bakes in optimal behavior after the first step.
The Bellman optimality equations therefore differ from the expectation equations in one key way: the
expectation over actions is replaced by a maximization, reflecting the agent's ability to choose.

## Solving Bellman optimality

The optimality equations include \(\max\) operators, e.g.

$$
V = \max_a \left(R^a + \gamma P^a V\right),
$$

where the max is applied component-wise across actions.  
This makes the system not linear in \(V\) (unlike the MRP case \((I-\gamma P)V=R\)). So, in general, there is no single matrix inverse expression like \((I-\gamma P)^{-1}R\).

### When is a closed form possible?

Closed form becomes possible once the \(\max\) is removed, which effectively happens when:

**(1) The optimal action is known in advance.**  
If we already know which action \(a^*(s)\) is optimal in each state, then the MDP reduces to the MRP induced by the deterministic policy \(\pi^*(s)=a^*(s)\):

$$
P^{\pi^*}(s'\mid s)=P(s'\mid s,a^*(s)),\qquad
R^{\pi^*}(s)=R(s,a^*(s)).
$$

Then

$$
V^* = V^{\pi^*} = (I-\gamma P^{\pi^*})^{-1} R^{\pi^*}.
$$

But notice: knowing \(a^*(s)\) is basically the problem we were trying to solve.

**(2) Tiny MDPs with simple structure.**  
For very small state spaces, we can solve piecewise: assume a maximizing action in each state, solve the corresponding linear system, then check whether the assumed actions are truly maximizing. This is feasible only for small problems because the number of action-combinations grows exponentially.

**(3) Special cases / restricted classes.**  
Some structured MDPs admit analytic solutions (certain deterministic shortest-path forms, special linear-quadratic control in continuous settings, etc.), but for general finite stochastic MDPs the standard approach is iterative computation.

### Iterative solution methods

**Model-based vs model-free**

- Model-based: uses \(P\) and \(R\) explicitly (value iteration, policy iteration, DP).
- Model-free: learns from sampled transitions without explicit \(P\) (Q-learning, SARSA).

**Value Iteration (VI)**

Value iteration applies the Bellman optimality backup repeatedly:

$$
V_{k+1}(s) \leftarrow
\max_a\left[
R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V_k(s')
\right].
$$

Intuition: repeatedly perform one-step lookahead improvements until the values stabilize.

**Policy Iteration (PI)**

Policy iteration alternates two steps:

1. Policy evaluation: compute \(V^{\pi}\) for the current policy \(\pi\) (this is linear, like an MRP),
2. Policy improvement: make the policy greedy w.r.t. current values:

$$
\pi_{\text{new}}(s)\in\arg\max_a\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^\pi(s')\right].
$$

Repeat until the policy stops changing (then it is optimal).

**Q-learning (off-policy, model-free)**

Q-learning aims to learn \(Q^*\) directly from samples \((S_t,A_t,R_{t+1},S_{t+1})\):

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t) + \alpha\Big(R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a')-Q(S_t,A_t)\Big).
$$

It uses a greedy max in the target, so it can learn the optimal greedy behavior even if the behavior policy explores.

**SARSA (on-policy, model-free)**

SARSA learns the value of the current behavior policy:

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t) + \alpha\Big(R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)\Big).
$$

Key difference: it uses the next action actually taken \((A_{t+1})\), rather than \(\max_{a'}\).

## Appendix

**Proof of Law of total expectation**

We want to show that:

$$
\mathbb{E}[X \mid Y] = \sum_{z} \Pr(Z=z \mid Y)\,\mathbb{E}[X \mid Z=z, Y].
$$

given \(Z\) is discrete (countable range) and \(\mathbb{E}[|X|]<\infty\). We prove the identity pointwise in \(y\). Fix any \(y\) with \(\Pr(Y=y)>0\). For simplicity we write conditional probabilities given \(Y=y\).

**Proof:**

First, by the definition of conditional expectation for a discrete \(X\),

$$
\mathbb{E}[X\mid Y=y]=\sum_{x} x\,\Pr(X=x\mid Y=y).
$$

Next, since the events \(\{Z=z\}\) form a disjoint partition, for each \(x\) we have

$$
\Pr(X=x\mid Y=y)=\sum_{z}\Pr(X=x, Z=z\mid Y=y).
$$

We now use the conditional product rule (chain rule). For events \(A,C,B\) with \(\Pr(B)>0\) and \(\Pr(C\cap B)>0\),

$$
\Pr(A,C\mid B)=\Pr(C\mid B)\Pr(A\mid C,B),
$$

because

$$
\Pr(C\mid B)\Pr(A\mid C,B)
=\frac{\Pr(C\cap B)}{\Pr(B)}\cdot \frac{\Pr(A\cap C\cap B)}{\Pr(C\cap B)}
=\frac{\Pr(A\cap C\cap B)}{\Pr(B)}
=\Pr(A\cap C\mid B).
$$

Apply this with \(A=\{X=x\}\), \(C=\{Z=z\}\), and \(B=\{Y=y\}\) to get

$$
\Pr(X=x, Z=z\mid Y=y)=\Pr(Z=z\mid Y=y)\,\Pr(X=x\mid Z=z, Y=y).
$$

Substituting into the previous expression gives

$$
\Pr(X=x\mid Y=y)=\sum_{z}\Pr(Z=z\mid Y=y)\,\Pr(X=x\mid Z=z, Y=y).
$$

Plug this into the definition of \(\mathbb{E}[X\mid Y=y]\):

$$
\mathbb{E}[X\mid Y=y]
=\sum_x x\left[\sum_{z}\Pr(Z=z\mid Y=y)\,\Pr(X=x\mid Z=z, Y=y)\right].
$$

Rearrange the sums (justified by integrability; e.g. by applying the argument to \(X^+\) and \(X^-\) separately):

$$
\mathbb{E}[X\mid Y=y]
=\sum_{z}\Pr(Z=z\mid Y=y)\left[\sum_x x\,\Pr(X=x\mid Z=z, Y=y)\right].
$$

The bracketed term is the definition of \(\mathbb{E}[X\mid Z=z, Y=y]\), so we conclude

$$
\mathbb{E}[X\mid Y=y]=\sum_{z}\Pr(Z=z\mid Y=y)\,\mathbb{E}[X\mid Z=z, Y=y].
$$

Since this holds for every \(y\) with \(\Pr(Y=y)>0\), replacing \(y\) by the random variable \(Y\) yields the desired identity

$$
\mathbb{E}[X\mid Y]=\sum_{z}\Pr(Z=z\mid Y)\,\mathbb{E}[X\mid Z=z, Y].
$$

# References

- https://github.com/zyxue/youtube_RL_course_by_David_Silver
