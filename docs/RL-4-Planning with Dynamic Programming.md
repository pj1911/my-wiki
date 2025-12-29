## Dynamic Programming

Dynamic programming (DP) is a method for solving optimization problems that unfold over stages (often interpreted as time). The term dynamic refers to this sequential structure: decisions made at one stage affect what is possible and valuable at later stages. The term programming is used in its classical sense from operations research: we are solving an optimization problem by searching over a space of decision rules. In many settings, those decision rules take the form of a policy, a rule that specifies what action to take in each state.

### Core idea
DP makes difficult problems tractable by exploiting structure. Rather than optimizing over the entire problem at once, it:

- decomposes the problem into smaller subproblems (typically corresponding to different stages),
- solves those subproblems,
- and then combines their solutions to construct a solution to the original problem.

The power of DP is not in the act of splitting alone, but in choosing a decomposition where the pieces can be solved efficiently and then reused.

## Principle of Optimality

The principle of optimality states that optimal behavior is self-consistent over time:
if we make an optimal decision now, then the remaining decisions must be optimal for the state that results.
Equivalently, an optimal solution can be decomposed into an optimal first step and an optimal continuation. Concretely, consider an optimal policy $\pi^*$. From any state $s$, its behavior can be viewed in two parts:

- the first action it takes in $s$, and
- the continuation policy it follows after transitioning to a successor state $s'$.

Since the continuation problem is the same decision problem but with a new starting state, optimality from $s$ implies optimality from the states that follow when $\pi^*$ is executed.

### How this enables DP
The principle of optimality is what makes a recursive decomposition feasible. It yields two structural advantages that DP exploits:

**Optimal substructure.**
Because the tail of an optimal solution must itself be optimal, we can solve the original problem by solving its subproblems.
The natural subproblem is: starting from a state $s$, what is the best achievable return from here onward?
Once we can answer that question for every state, selecting an optimal first action becomes a local comparison of alternatives, each evaluated using the value of an optimal continuation.

**Overlapping subproblems.**
Many different histories can lead to the same state, meaning the same continuation problem appears repeatedly.
DP avoids recomputing these repeated subproblems by storing their solutions and reusing them.

### MDPs as a DP setting
Markov Decision Processes fit this framework exactly. In an MDP, the “situation we are in” can be summarized by the current state $s$ (the Markov property), so the relevant subproblem is well-defined: optimize future return starting from $s$.
The value function is therefore a direct representation of subproblem solutions:


$$
v^*(s) \;=\; \text{optimal expected return starting from state } s.
$$


The principle of optimality then justifies evaluating actions by combining immediate outcomes with the value of optimal continuation, which leads to the Bellman viewpoint that underlies planning algorithms.

### A consistency consequence along reachable states
A useful implication is that optimality must persist along the trajectories induced by an optimal policy.
If a policy $\pi$ is optimal from a state $s$ in the sense that


$$
v_{\pi}(s)=v^{*}(s),
$$


then it cannot “fall behind” after the first transition. In particular, for any successor state $s'$ that can be reached from $s$ when following $\pi$, the policy must also achieve optimal value:


$$
v_{\pi}(s')=v^{*}(s') \qquad \text{for every successor state $s'$ reachable from $s$ under $\pi$.}
$$


Informally: if we are optimal now, we must remain optimal on every continuation. This is exactly what Bellman-style updates enforce across the state space. With this principle in place, we can now turn to planning in an MDP and see how DP implements these Bellman updates in practice.

## Planning by Dynamic Programming in an MDP

In reinforcement learning, planning means computing decisions using a model of the environment, rather than updating solely from real experience.
Concretely, a planning method takes as input an MDP model, for example the states, actions, a transition model, a reward model, and a discount factor and performs computation (often offline, or via simulated lookahead) to produce a decision rule, i.e., a policy. Dynamic Programming (DP) is the classical planning approach for MDPs when this model is available. Assuming we know all the model parameters ($\mathcal{S}$, $\mathcal{A}$, $P_{ss'}^{a}$, $R_{s}^{a}$, $\gamma$), DP computes long-term consequences by exploiting Bellman recursions. These recursions express the value of a state (or state--action pair) in terms of the immediate reward and the discounted value of successor states, allowing DP algorithms to improve value estimates and policies through repeated one-step lookahead updates. Within this planning setting, DP algorithms are typically organized around two complementary tasks: prediction and control, which we define precisely next.

### Prediction vs. control
It is useful to distinguish two closely related computational goals when we plan with a known MDP model:

- Prediction: the policy $\pi$ is fixed. The task is to evaluate it by computing its value function $v_{\pi}$, which gives the expected discounted return from each state when actions are chosen according to $\pi$.
- Control: the policy is not fixed. The task is to find the optimal value function $v^{*}$ and an optimal policy $\pi^{*}$ that achieves it.

These goals correspond to different Bellman operators. Prediction uses Bellman expectation updates, which evaluate a given policy via model-based expectations, whereas control uses Bellman optimality updates, which push values toward the best achievable behavior and thereby support improvement toward an optimal policy which is the ultimate goal of planning.

### Iterative policy evaluation (prediction)
Assume we are given a policy $\pi$ and asked: "If we follow $\pi$, what long-term return should we expect from each state?" DP answers this by iteratively refining an estimate of $v_{\pi}$ using the Bellman expectation equation. One simple procedure is to begin with an initial guess, for instance


$$
v_{1}(s)=0 \qquad \text{for all } s\in\mathcal{S},
$$


and then repeatedly apply a one-step lookahead update. At iteration $k+1$, for every state $s$,


$$
v_{k+1}(s)=\sum_{a\in\mathcal{A}}\pi(a\mid s)\Bigl(R_{s}^{a}+\gamma\sum_{s'\in\mathcal{S}}P_{ss'}^{a}\,v_{k}(s')\Bigr).
$$


This update has a direct interpretation: the value of a state equals the expected immediate reward plus the discounted value of the next state, where the expectation is taken over the action choice under $\pi$ and the environment dynamics. Repeating this update causes $v_k$ to approach the true value function $v_{\pi}$.

### Synchronous backups (how the updates are applied)
We now describe a standard way to apply the update rule used in iterative policy evaluation in DP.

**Synchronous backups.**
At iteration $k+1$, we perform a full sweep over the state space and update every state using only the values from the previous iterate:


$$
\forall s\in\mathcal{S}:\quad v_{k+1}(s)\;\leftarrow\;\mathcal{T}^{\pi}v_k(s),
$$


where $\mathcal{T}^{\pi}$ is the Bellman expectation operator.

It is worth emphasizing, in DP planning, the value function is primarily a tool for improvement: once we have an estimate of how good each state is under the current policy, we can use the model to ask a sharper question, namely whether another action would lead to a higher expected return. This question will be answered in the next section.

## Policy Iteration

Up to now, we have used DP for prediction: given a fixed policy $\pi$, compute its value function $v_\pi$.
We now turn to the central goal in an MDP: control, which answers the question of finding the best policy.

The main idea behind policy iteration is simple: values enable improvement.
If $v_\pi(s)$ tells us the expected long-term return from each state when following $\pi$, then we can use the model to ask,
in every state, whether some other action would lead to a higher expected return.
Policy iteration formalizes this as an alternating loop of evaluation and improvement.

### The two-step loop
Policy iteration maintains a sequence of policies $\pi_0,\pi_1,\pi_2,\dots$ and alternates between:

1. Policy evaluation: given the current policy $\pi_k$, compute (or approximate) its value function $v_{\pi_k}$, typically by repeated Bellman expectation backups (to near convergence).
2. Policy improvement: construct a new (often deterministic) policy by acting greedily with respect to $v_{\pi_k}$:


$$
\pi_{k+1}(s)\in \arg\max_{a\in\mathcal{A}}
\Bigl(R_s^a+\gamma\sum_{s'\in\mathcal{S}}P_{ss'}^a\,v_{\pi_k}(s')\Bigr).
$$


In other words: estimate how good our current policy is, then switch (state-by-state) to the action that looks best under that estimate, and repeat.

### Policy improvement: why greedy is safe
To make the improvement step precise, it is convenient to introduce the action-value function under $\pi$:


$$
q_{\pi}(s,a)=\mathbb{E}_{\pi}\!\left[\,R_{t+1}+\gamma v_{\pi}(S_{t+1}) \;\middle|\; S_t=s,\;A_t=a\,\right].
$$


This quantity means: take action $a$ now, then follow $\pi$ thereafter.

**Greedy improvement.**
Given $q_\pi$, define a new policy $\pi'$ that chooses, in each state, an action maximizing $q_\pi$:


$$
\pi'(s)\in \arg\max_{a\in\mathcal{A}} q_{\pi}(s,a).
$$


Because $\pi'(s)$ is a maximizer, we immediately have, for every state $s$,


$$
q_{\pi}\bigl(s,\pi'(s)\bigr)=\max_{a\in\mathcal{A}}q_{\pi}(s,a)\;\ge\; q_{\pi}\bigl(s,\pi(s)\bigr)=v_{\pi}(s).
$$


Read this as a one-step statement:

- $q_{\pi}(s,\pi(s))$ is "do what $\pi$ would do now, then keep following $\pi$",
- $q_{\pi}(s,\pi'(s))$ is "do the greedy action now, then keep following $\pi$",
- greedy cannot be worse than $\pi$'s own action under the same continuation $\pi$.

**From one step to the full return.**
The key point is that this one-step improvement can be unrolled over time.
Starting from


$$
v_{\pi}(s)\le q_{\pi}\bigl(s,\pi'(s)\bigr)
=\mathbb{E}_{\pi'}\!\left[\,R_{t+1}+\gamma v_{\pi}(S_{t+1}) \;\middle|\; S_t=s\,\right],
$$


We apply the same greedy argument at subsequent states, repeatedly bounding
$v_\pi(S_{t+1})$, then $v_\pi(S_{t+2})$, and so on.


$$
\begin{aligned}
v_{\pi}(s)
&\le q_{\pi}\bigl(s,\pi'(s)\bigr)
= \mathbb{E}_{\pi'}\!\left[\,R_{t+1}+\gamma v_{\pi}(S_{t+1}) \;\middle|\; S_t=s\,\right] \\
&\le \mathbb{E}_{\pi'}\!\left[\,R_{t+1}+\gamma q_{\pi}\bigl(S_{t+1},\pi'(S_{t+1})\bigr) \;\middle|\; S_t=s\,\right] \\
&\le \mathbb{E}_{\pi'}\!\left[\,R_{t+1}+\gamma R_{t+2}
+\gamma^{2} q_{\pi}\bigl(S_{t+2},\pi'(S_{t+2})\bigr) \;\middle|\; S_t=s\,\right] \\
&\le \cdots \\
&\le \mathbb{E}_{\pi'}\!\left[\,R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{n-1}R_{t+n}
+\gamma^{n} v_{\pi}(S_{t+n}) \;\middle|\; S_t=s\,\right] \\
&\le \mathbb{E}_{\pi'}\!\left[\,R_{t+1}+\gamma R_{t+2}+\cdots \;\middle|\; S_t=s\,\right]
= v_{\pi'}(s).
\end{aligned}
$$


Taking the limit as $n\to\infty$ (the residual term vanishes under discounting), we obtain the policy improvement guarantee:


$$
v_{\pi'}(s)\;\ge\; v_{\pi}(s)\qquad \forall s\in\mathcal{S}.
$$


**Takeaway.**
Greedy policy improvement is monotone: it never makes the policy worse.
At worst, it leaves values unchanged, otherwise, it strictly improves them in at least one state.

### Why policy iteration stops at an optimal policy?
If the greedy improvement step leaves the policy unchanged, then policy iteration has reached a fixed point (and we can regard the algorithm as having converged). Equivalently, for every state $s\in\mathcal{S}$,


$$
\pi(s)\in \arg\max_{a\in\mathcal{A}} q_{\pi}(s,a)
\quad\Longrightarrow\quad
q_{\pi}\bigl(s,\pi(s)\bigr)=\max_{a\in\mathcal{A}} q_{\pi}(s,a).
$$


But $q_{\pi}(s,\pi(s))=v_{\pi}(s)$ by definition, so we have


$$
v_{\pi}(s)=\max_{a\in\mathcal{A}} q_{\pi}(s,a)\qquad \forall s\in\mathcal{S}.
$$


This is exactly the Bellman optimality condition: in each state, the value equals the best achievable one-step lookahead.
A policy that is greedy with respect to its own value function is therefore optimal, which implies


$$
v_{\pi}(s)=v^{*}(s)\qquad \forall s\in\mathcal{S},
\qquad\text{and}\qquad
\pi=\pi^{*}.
$$


**Intuition.**
The algorithm stops only when there is no state in which a different action would look better under the policy's own long-term value estimates. At that point the policy already satisfies the optimality equations, so it must be optimal.

### Is this a greedy algorithm?
Policy iteration does include a greedy step, but it is not greedy in the short-sighted sense of maximizing immediate reward.

**Greedy with respect to long-term return.**
The improvement step chooses actions using the action-value under the current policy,


$$
q_{\pi}(s,a)=\mathbb{E}\!\left[\,R_{t+1}+\gamma v_{\pi}(S_{t+1}) \mid S_t=s,\;A_t=a\,\right],
$$


so actions are compared by immediate reward plus discounted future value. This is fundamentally different from the myopic rule


$$
a=\arg\max_a \mathbb{E}[R_{t+1}\mid s,a],
$$


which ignores the effect of actions on future states.

## Modified Policy Iteration

Classic policy iteration is conceptually elegant: evaluate the current policy $\pi_k$ until we have its exact value function $v_{\pi_k}$, then improve greedily to obtain $\pi_{k+1}$. But this clean separation can be computationally wasteful. Full policy evaluation may require many sweeps over the state space, even though the very next step will replace the policy anyway. This raises a natural question:
how accurately do we need to evaluate the current policy before improving it?

### Truncating policy evaluation
Modified policy iteration answers by relaxing the evaluation step. Instead of computing $v_{\pi_k}$ to convergence, we perform only a limited amount of evaluation, producing an approximation $\tilde v_k$. Two common truncation choices are:

- stop iterative evaluation once successive value estimates change by less than a tolerance $\varepsilon$ (an $\varepsilon$-stopping rule), or
- run a fixed number of sweeps, say $m$, of the Bellman expectation update.

After this partial evaluation, we perform the same greedy improvement step as in policy iteration, replacing $\pi_k$ by a policy that is greedy with respect to the current estimate (whether exact or approximate).

**Why this can still be effective.**
The improvement step does not require a perfectly accurate value function, it requires a value estimate that is good enough to guide better action choices.
If $\tilde v_k$ already captures the broad shape of long-term returns, then a greedy improvement step often corrects the most obvious suboptimal action choices immediately.
As a result, repeatedly doing "some evaluation + improvement" can reach a strong policy using fewer total sweeps than insisting on "perfect evaluation + improvement" at every iteration.

### A spectrum: policy iteration $\rightarrow$ value iteration
Seen through this lens, policy iteration and value iteration are not fundamentally different algorithms, but endpoints of a continuum determined by how much work we invest in evaluation before improving:

- Policy iteration: evaluate to convergence, then improve.
- Modified policy iteration: evaluate partially, then improve.
- Value iteration: perform only a single Bellman-style sweep and immediately improve.

In value iteration, evaluation and improvement effectively fuse into a single update that pushes values directly toward optimality. This is the key intuition behind value iteration, which we develop in the next section.

## Value Iteration and its relation to Policy Iteration

Value iteration is a dynamic programming method for the control problem: it aims to compute the optimal value function $v^*$ and, from it, an optimal policy $\pi^*$. It is based on the Bellman optimality equation,


$$
v^{*}(s)=\max_{a\in\mathcal{A}}
\Bigl(R_s^a+\gamma\sum_{s'\in\mathcal{S}}P_{ss'}^a\,v^{*}(s')\Bigr),
$$


which states that the optimal value of a state equals the best one-step lookahead, immediate reward plus discounted optimal continuation value.

### The value iteration update
Because $v^*$ is unknown, value iteration begins from an arbitrary initial guess $v_0$ (often $v_0\equiv 0$) and repeatedly applies the Bellman optimality backup:


$$
v_{k+1}(s)\;=\;\max_{a\in\mathcal{A}}
\Bigl(R_s^a+\gamma\sum_{s'\in\mathcal{S}}P_{ss'}^a\,v_k(s')\Bigr),
\qquad \forall s\in\mathcal{S}.
$$


This update has the same form as the Bellman optimality equation, except that it uses the current estimate $v_k$ on the right-hand side rather than the unknown $v^{*}$. In operator notation, it is simply $v_{k+1} = \mathcal{T}^* v_k $
where $\mathcal{T}^*$ is the Bellman optimality operator. The optimal value function $v^{*}$ is the unique fixed point of this operator (in the discounted, finite setting), meaning it satisfies $v^{*}=\mathcal{T}^* v^{*}$. Value iteration repeatedly applies $\mathcal{T}^*$; if the iterates converge to some limit $v_\infty$, then necessarily $v_\infty$ is a fixed point and hence $v_\infty=v^{*}$.

### How to obtain a policy
Value iteration updates only values, it does not need to store a policy during the updates. Once a value estimate $v_k$ is available, we can extract a greedy policy by one-step lookahead:


$$
\pi_k^{\text{greedy}}(s)\in\arg\max_{a\in\mathcal{A}}
\Bigl(R_s^a+\gamma\sum_{s'\in\mathcal{S}}P_{ss'}^a\,v_k(s')\Bigr).
$$


In practice, one may compute this greedy policy only at the end (to obtain $\pi^*$ from $v^*$), or track it along the way to see how behavior is improving.

### How value iteration differs from policy iteration
Policy iteration makes the policy explicit and alternates two operations:


$$
\pi_k \;\xrightarrow{\ \text{evaluate}\ }\; v_{\pi_k}
\;\xrightarrow{\ \text{improve}\ }\; \pi_{k+1},
$$


i.e., it evaluates a fixed policy using the Bellman expectation equation and then improves it greedily.

Value iteration skips the explicit evaluation of a fixed policy. Instead, each sweep applies the Bellman optimality backup directly to $v$, so the "improvement" idea is built into the update itself:


$$
v_k \;\xrightarrow{\ \text{optimality backup}\ }\; v_{k+1}.
$$


As a consequence, intermediate value functions $v_k$ are best viewed as improving approximations heading toward $v^*$, they need not equal $v_\pi$ for any single stationary policy $\pi$ at that iteration.

## Synchronous Dynamic Programming Algorithms

All of the dynamic programming methods in this chapter rely on the same basic operation:
use the model to look one step ahead and update a value estimate.
Where they differ is in the objective, prediction versus control, and therefore in which Bellman relationship drives the update. In the tabular setting, these differences lead to three canonical algorithms.

1. Prediction $\rightarrow$ Iterative Policy Evaluation

   Goal: evaluate a \underline{fixed} policy $\pi$ by computing its value function $v_{\pi}$.

   Bellman relationship: the Bellman expectation equation, which averages over the actions selected by $\pi$ (and over next states under the dynamics).

   Algorithmic pattern: repeatedly update $v(s)$ using the expected immediate reward plus the discounted expected value of successor states under $\pi$, until the values are consistent with following $\pi$.

2. Control $\rightarrow$ Policy Iteration

   Goal: find an optimal policy $\pi^{*}$.

   Bellman relationships:
   - use the Bellman expectation equation to (approximately or exactly) evaluate the current policy, and
   - apply a greedy improvement step to update the policy.

   Algorithmic pattern: alternate between (i) evaluating the current $\pi$ and (ii) improving it by choosing, in each state, an action that maximizes one-step lookahead using the current value estimate. The resulting sequence of policies is monotone: it never gets worse.

3. Control $\rightarrow$ Value Iteration

   Goal: compute the optimal value function $v^{*}$ and then extract an optimal policy from it.

   Bellman relationship: the Bellman optimality equation, which takes a max over actions.

   Algorithmic pattern: repeatedly apply the optimality backup directly to $v$. Informally, each sweep performs an "improve everywhere" step, without explicitly storing a policy during the updates (though a greedy policy can be extracted at any time).

### Complexity analysis
In their simplest tabular forms, these DP planning methods store state values:


$$
v_{\pi}(s)\ \text{for prediction},\qquad v^{*}(s)\ \text{for control}.
$$


Let $n=\lvert\mathcal{S}\rvert$ and $m=\lvert\mathcal{A}\rvert$. A synchronous sweep computes one backup for every stored entry.

#### State-value backups $v(s)$
In tabular DP control (e.g., value iteration), the backup is


$$
(\mathcal{T}^* v)(s)
=\max_{a\in\mathcal{A}}\Bigl(R_s^a+\gamma\sum_{s'\in\mathcal{S}}P_{ss'}^a\,v(s')\Bigr).
$$


For a fixed state $s$, the work in this formula is:

- the $\max_{a\in\mathcal{A}}$ loops over $m$ actions, and
- for each action, the inner $\sum_{s'\in\mathcal{S}}$ loops over $n$ successor states.

So one state backup costs $O(mn)$. A full sweep updates all $n$ states, hence


$$
\underbrace{n}_{\text{states}}
\;\times\;
\underbrace{m}_{\text{actions per state}}
\;\times\;
\underbrace{n}_{\text{successors per action}}
\;=\;
O(mn^2)
\quad \text{per sweep.}
$$


#### Action-value backups $q(s,a)$
If we store action-values, there are $mn$ entries. A common control-style backup is


$$
(\mathcal{T}^* q)(s,a)
=R_s^a+\gamma\sum_{s'\in\mathcal{S}}P_{ss'}^a\,\max_{a'\in\mathcal{A}} q(s',a').
$$


For a fixed pair $(s,a)$, the work in this formula is:

- the $\sum_{s'\in\mathcal{S}}$ loops over $n$ successor states, and
- inside the sum, the $\max_{a'\in\mathcal{A}}$ loops over $m$ actions.

So one $(s,a)$ backup costs $O(nm)$. A full sweep updates all $mn$ pairs, hence


$$
\underbrace{(mn)}_{\text{state--action pairs}}
\;\times\;
\underbrace{n}_{\text{successors}}
\;\times\;
\underbrace{m}_{\text{actions in }\max}
\;=\;
O(m^2n^2)
\quad \text{per sweep.}
$$


**Takeaway**
All synchronous DP methods are one-step lookahead sweeps, they differ mainly in the Bellman operator (expectation for prediction, optimality for control) and in whether a policy is maintained explicitly (policy iteration) or extracted from values (value iteration).

## Asynchronous Dynamic Programming

Up to this point, we have treated DP updates as synchronous: at iteration $k$ we compute a complete new table $v_{k+1}$ from $v_k$ by performing a full sweep over all states. This viewpoint is clean and easy to analyze, but it can be wasteful in practice: many states may already be nearly correct, while a small set of states may still have large errors. Synchronous sweeps spend equal effort everywhere, regardless of where the value function most needs improvement.

Asynchronous DP keeps the same Bellman backup, but changes how it is applied. Instead of updating every state on every iteration, it updates one (or a few) states at a time and immediately overwrites the stored value:


$$
v(s)\leftarrow (\mathcal{T}v)(s).
$$


Because updates are not forced to occur uniformly, asynchronous methods can:

- avoid recomputing values for states that are already accurate,
- focus computation where the current approximation is most wrong or most relevant,
- propagate new information faster (since later updates can use earlier updated values).

In discounted finite MDPs, asynchronous DP still converges as long as every state is updated infinitely often, but it often reaches a useful approximation in far fewer backups than full sweeps.

### Three useful versions of asynchronous DP

#### In-place dynamic programming (faster propagation, less memory)
A minimal change from synchronous DP is to switch from a two-table update to an in-place update.
In synchronous value iteration we conceptually separate "read" and "write" tables:


$$
v_{\text{new}}(s)\leftarrow \max_{a\in\mathcal{A}}
\left(\mathcal{R}_s^a + \gamma \sum_{s'\in\mathcal{S}}\mathcal{P}_{ss'}^a\,v_{\text{old}}(s')\right),
\qquad
v_{\text{old}} \leftarrow v_{\text{new}}.
$$


This ensures each backup uses only stale values from iteration $k$.

In in-place value iteration we maintain a single table $v$ and overwrite entries immediately:


$$
v(s)\leftarrow \max_{a\in\mathcal{A}}
\left(\mathcal{R}_s^a + \gamma \sum_{s'\in\mathcal{S}}\mathcal{P}_{ss'}^a\,v(s')\right).
$$


The benefit is that information can move through the state space more quickly: later backups in the same sweep can already exploit improvements made earlier. Practically, in-place updates often reduce the number of sweeps needed to achieve a given accuracy, and they also avoid storing a second full table.

#### Prioritised sweeping (spend backups where they matter most)
Synchronous sweeps spend the same effort on every state, even though at a given moment some states violate the Bellman equation much more than others (i.e., their current values are much farther from their one-step backup). Prioritised sweeping takes advantage of this by choosing the next state to update based on its current Bellman error:

$$
\left|
\max_{a\in\mathcal{A}}
\left(\mathcal{R}_s^a + \gamma \sum_{s'\in\mathcal{S}}\mathcal{P}_{ss'}^a\,v(s')\right)
- v(s)
\right|
$$

Intuitively, a large error indicates that the current value at $s$ is far from what the Bellman equation demands, so updating $s$ is likely to produce a meaningful improvement. A typical implementation of this would be:

- selects the state with the largest current error and performs its backup,
- then updates priorities for states likely to be affected next (often predecessor states),
- and uses a priority queue to make this selection efficient.

Compared to full sweeps, prioritised sweeping can reach an accurate value function with far fewer backups when errors are localized, because it concentrates computation on the “hard” parts of the problem.

#### Real-time dynamic programming (update only states you actually encounter)
Sometimes the goal is not to make the value function accurate everywhere, but to make it accurate where the agent will actually go. In large MDPs, many states may be irrelevant to near-term decision-making from the current start state. Real-time DP exploits this by backing up only states encountered along simulated (or real) trajectories.

After taking a step at time $t$, we update the current state $S_t$:


$$
v(S_t)\leftarrow \max_{a\in\mathcal{A}}
\left(\mathcal{R}_{S_t}^a + \gamma \sum_{s'\in\mathcal{S}}\mathcal{P}_{S_t s'}^a\,v(s')\right).
$$


This can be dramatically cheaper than sweeping when the reachable region is a small fraction of the full state space. The tradeoff is that states that are never visited will never be improved—but if they are irrelevant to the task from the current start distribution, this “selective accuracy” is exactly what we want.

**Takeaway.**
Synchronous DP spends a fixed budget per sweep by updating everything.
Asynchronous DP spends a flexible budget by updating selected states, often achieving good value estimates and policies with substantially less total computation.

## Approximate Dynamic Programming

Tabular DP assumes we can store and update a distinct number for every state $s\in\mathcal{S}$. When $\lvert\mathcal{S}\rvert$ is enormous (or continuous), that assumption breaks down. Approximate DP keeps the DP logic, Bellman backups and repeated improvement, but replaces the value table with a function approximator.

### Replace the table with a function class
Instead of representing a value function as $v(s)$ for each state, we represent it by a parameterized approximation

$$
\hat v(s,\mathbf{w}),
$$

where $\mathbf{w}$ denotes the parameters (weights). The goal is that a single parameter vector generalizes across many states, allowing compact storage and sharing statistical strength.

### DP directly on the approximation (Fitted Value Iteration)
With function approximation, we can no longer apply the Bellman operator to every state and store the result exactly. A standard workaround is to run DP in a sampled and projected way: we apply a Bellman backup on a set of sampled states, then fit our approximator to match those backed-up values. This template is known as Fitted Value Iteration (FVI).

**Setup and initialization.**
We assume an MDP with discount factor $\gamma$ and access to a generative model (or known dynamics) so that, for any queried pair $(s,a)$, we can obtain the immediate reward and either compute or estimate the expected next-state value.
Our goal is to approximate the optimal value function $v^*$, but since a tabular representation is infeasible, we restrict ourselves to a parameterized function class


$$
\{\hat v(\cdot,\mathbf{w}) : \mathbf{w}\in\mathbb{R}^d\},
$$


(e.g., linear features or a neural network) and seek parameters $\mathbf{w}$ such that $\hat v(\cdot,\mathbf{w})\approx v^*$ over the states of interest. To train this approximation we also choose a state-sampling scheme, modeled as sampling $s\sim \mu$ for some distribution $\mu$ over $\mathcal{S}$ (uniform over a region, concentrated near a start-state distribution, or induced by trajectories under a behavior policy). Finally, we pick an initial parameter vector $\mathbf{w}_0$ (commonly $\hat v(\cdot,\mathbf{w}_0)\equiv 0$ or small random weights) and then iteratively improve $\hat v$ using Bellman targets and regression.

**Iteration.**
For $k=0,1,2,\ldots$ repeat:

1. Sample states and form Bellman targets.

   Sample a finite training set of states $\tilde{\mathcal{S}}_k=\{s^{(1)},\dots,s^{(n)}\}$ from $\mu$.
   For each sampled state $s\in\tilde{\mathcal{S}}_k$, compute a one-step optimality target by backing up the current approximation:


$$
\tilde v_k(s)=
\max_{a\in\mathcal{A}}
\left(
  \mathcal{R}_s^a
  +\gamma\sum_{s'\in\mathcal{S}}\mathcal{P}_{ss'}^a\,\hat v(s',\mathbf{w}_k)
\right).
$$


   (If we only have a simulator rather than an explicit transition matrix, the expectation over $s'$ can be approximated by samples.)

2. Fit the approximator to the targets (projection step).

   Update parameters by supervised regression so that the next approximation matches these targets on the sampled states:


$$
\mathbf{w}_{k+1}
\in
\arg\min_{\mathbf{w}}
\sum_{s\in\tilde{\mathcal{S}}_k}
\big(\hat v(s,\mathbf{w})-\tilde v_k(s)\big)^2.
$$


   Equivalently, $\hat v(\cdot,\mathbf{w}_{k+1})$ is trained on the dataset
   $\{(s,\tilde v_k(s)):\ s\in\tilde{\mathcal{S}}_k\}$.

**Interpretation.**
Each iteration performs two conceptual steps: a Bellman improvement step (create targets using one-step lookahead) followed by a projection step (compress the backed-up values back into the function class by regression). This recovers the DP idea of repeated Bellman updates, but replaces exact tabular storage with approximation and generalization.

## References

- https://github.com/zyxue/youtube_RL_course_by_David_Silver
