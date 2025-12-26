# Introduction to Reinforcement Learning

Reinforcement Learning (RL) is the science of decision making: an agent learns what to do by interacting with an environment, trying to maximize it's reward over time. What makes RL different from many other machine learning methods is the kind of feedback it receives. Instead of a supervisor telling the correct answer, the agent only sees a reward signal, and that signal may be delayed, meaning the effect of an action might show up much later. Time is therefore central: the data comes as a sequence, not as independent samples (so it is not i.i.d.), and the agent's actions actively influence what it experiences next. Typical examples of RL includes learning to play games (like Atari or Chess), training robots to walk or manipulate objects, optimizing long-term choices in recommender systems, and controlling real-world systems such as traffic lights, resource allocation, or scheduling.

## Motivation for winning: Rewards and Reward Hypothesis

A natural next step is to talk about rewards. In RL, the reward is usually written as \(R_t\). It is a scalar value that acts as a feedback signal for the agent at time step \(t\). The agent's job is to maximize the cumulative (over time) reward.

This naturally motivates the reward hypothesis: all goals can be described as maximizing the expected cumulative reward. This is a very strong assumption because it says that any notion of a goal can always be turned into an objective of maximizing the expected cumulative reward. But this assumption may not always hold: many goals are hard to compress into a single number. Many real objectives often include multiple preferences, safety constraints, fairness considerations, or instructions like do not do X,. These can be awkward to express with one reward signal. When the reward is an imperfect proxy, an agent may optimize the number while missing the real intent.

For example, consider an autonomous car with the goal get to the destination quickly, but never endanger pedestrians, obey traffic rules, and keep passengers comfortable. Speed, safety, legality, and comfort can conflict, and turning all of that into one reward number is tricky. A poorly chosen reward might push the car to drive aggressively to gain time, or to exploit loopholes in the definition of safe, which shows why reducing a rich goal to a single reward can be fragile.

In application, when the real objective is complex, RL still forces a simplification: the agent must ultimately make decisions by comparing actions using a single objective. Even if we care about many things at once (speed, safety, cost, comfort), the standard RL view is that these preferences must be converted into one scalar reward signal. In practice, the hard part is not the optimization itself, but choosing a reward function that correctly represents what we actually want, since any mismatch can push the agent to optimize the number rather than the intended goal.

### Sequential Decision Making (a Unifying Framework)

The key observation from the above examples is that RL problems are rarely one-shot decisions. They are sequential: an agent repeatedly chooses actions, those actions can have long-term consequences, and rewards can arrive later. Because of this, it can be optimal to give up an immediate reward if it leads to a larger total reward in the future.

This is where the unifying framework comes in. RL models tasks as sequential decision-making problems where, at each time step \(t\), the agent observes the situation, takes an action, and receives a reward \(R_t\). The objective is not to maximize \(R_t\) at a single step, but to maximize the expected total future reward accumulated over time. This viewpoint lets very different problems share the same mathematical structure: investing money (pay a cost now, gain later), or blocking an opponent in a game (sacrifice a move now to improve winning chances later). In the next part, we will formalize this idea with the standard RL model used to describe the environment and the agent's interaction with it.

## Agent-Environment Interaction

RL starts with a simple setup: an agent interacts with an environment over time. The agent is the decision-maker we want to control (think of it as the brain), and the environment is everything outside the agent that reacts to its actions. Note, the agent does not directly control the environment.

As a sequential decision process, let time is split into steps \(t = 1,2,3,\dots\). At each step \(t\):
the agent receives an observation \(O_t\) (what it can currently sense), uses it to choose an action \(A_t\), and then the environment responds by producing a reward and the next observation. A common way to write this is:

$$
O_t \xrightarrow{\text{agent chooses}} A_t \xrightarrow{\text{environment responds}} (R_{t+1}, O_{t+1}).
$$

Then, \(O_{t+1}\) becomes the input for the next decision process. This loop repeats, so decisions can have long-term effects through how they change future observations and rewards. Over time, these interactions produce a history (full time-ordered record) of what the agent has observed, done, and received so far. One way to write the history at time \(t\) is

$$
H_t = (O_1, R_1, A_1, \dots, A_{t-1}, O_t, R_t),
$$

which we can think of as all observable variables up to time \(t\). This is the only information the agent can directly use. The environment may contain many hidden variables (for example, internal physics, other agents' intentions, or unobserved noise) that affect what happens next, but the agent does not get to see them. A completely general agent could choose actions as a function of the entire history, i.e., a decision rule of the form \(A_t = \pi(H_t)\). The problem is that histories grow longer over time, which makes them inconvenient to store and reason about.

This motivates the idea of a state: a compressed summary of the history that keeps the information needed to decide what to do next. Formally, we define the state at time \(t\) as a function of history,

$$
S_t = f(H_t),
$$

so the agent can act using \(S_t\) instead of the full \(H_t\). The main goal is to choose \(f\) so that \(S_t\) captures what matters the most for predicting the future, while remaining much smaller and easier to work with than the entire history.

### State

The word state is overloaded in RL, and it helps to separate three related ideas.

- First, the environment state \(S_t^{e}\) is the environment's private description of the world: whatever internal variables it uses to generate the next observation and reward. The agent typically cannot see \(S_t^{e}\) directly. Even if it could, parts of it might be irrelevant for decision making, so having access to the full environment state is not always necessary.
- Second, the agent state \(S_t^{a}\) is the agent's internal representation of what is going on. This is what the agent actually stores and updates while interacting with the environment. In general, it can be any function of the observable history,

$$
S_t^{a} = f(H_t),
$$

and actions are chosen based on this internal state. This is usually the most practical notion of state in application, because it is under the agent's control.
- Third, an information state (or Markov state) is a special kind of state that contains all useful information from the history for predicting the future. One common way to express this is

$$
\mathbb{P}(S_{t+1}\mid S_t) = \mathbb{P}(S_{t+1}\mid S_1,\dots,S_t),
$$

which is the Markov property. When a state is Markov, we can treat it as a sufficient summary: in principle, we can throw away the full history and still act optimally using only \(S_t\).

### Fully Observable Environments (Assumption)

For most of our discussion, we will focus on the simplest and most common setting: fully observable environments. Full observability means the agent directly observes the underlying state of the environment, so there is no hidden information from the agent's point of view. In this case, the observation is the state:

$$
O_t = S_t.
$$

Because the agent can see the full state, we can treat the agent state, environment state, and information (Markov) state as the same object:

$$
O_t = S_t^{a} = S_t^{e}.
$$

This is important because it makes the problem much easier to model and solve: the current state contains all the information needed to predict what happens next, so we do not need to carry the entire history. Formally, this setting is called a Markov decision process (MDP), and it will be the main framework used in the discussions that follow.

### Partially Observable Environments (Reality)

In many realistic settings, the agent cannot directly observe the true environment state. This is called partial observability: the agent only gets an observation signal that provides an indirect view of the environment. Typical examples are a robot with a camera that is not told its absolute location or a trading agent that only observes current prices not the trends.

In this setting, the environment has an internal (often hidden) state \(S_t^{e}\), but the agent does not observe it directly. Instead, it receives observations \(O_t\). As a result, the agent's internal state is generally not equal to the environment state:

$$
S_t^{a} \neq S_t^{e}.
$$

Formally, this setup is modeled as a partially observable Markov decision process (POMDP). Even if the environment dynamics are Markov in \(S_t^{e}\), the agent only sees \(O_t\), so it must build its own state representation to decide well.

A few common choices for the agent state \(S_t^{a}\) are:

(1) Complete history:

$$
 S_t^{a} = H_t,
$$

where \(H_t\) is the full sequence of observations, actions, and rewards up to time \(t\).

(2) Belief over environment states:

$$
S_t^{a} = \big(\mathbb{P}[S_t^{e}=s^1], \ldots, \mathbb{P}[S_t^{e}=s^n]\big),
$$

which is a probability distribution over possible environment states. This is often called the belief state, and it summarizes uncertainty about what the true hidden state might be.

(3) Learned memory (recurrent neural network)

$$
S_t^{a} = \sigma\!\left(S_{t-1}^{a} W_s + O_t W_o\right),
$$

where \(\sigma(\cdot)\) is a nonlinear function and \(W_s, W_o\) are parameters. Here the agent compresses the past into a compact internal representation that is updated every step.

All three approaches aim for the same goal: construct an internal state \(S_t^{a}\) that contains enough information from the history to choose good actions, even when the true environment state is not directly observable.

## Major Components of an RL Agent

An RL agent is usually described in terms of a few core components that play different roles in decision making. Depending on the algorithm, an agent may use one or more of the following: a policy, a value function, and a model of the environment. Different RL methods emphasize different components, but these ideas form a common language across the field.

### Policy

A policy describes the agent’s behavior. It specifies how the agent chooses actions based on the current state. In other words, it is a mapping from states to actions. In the simplest case, the policy is deterministic, meaning it always picks the same action in a given state:

$$
a = \pi(s).
$$

More generally, policies are stochastic. A stochastic policy defines a probability distribution over actions given a state:

$$
\pi(a \mid s) = \mathbb{P}[A_t = a \mid S_t = s].
$$

Stochastic policies are useful when the environment is uncertain, when exploration is needed, or when randomization itself is beneficial.

### Value Functions

Value functions predict future reward and are used to evaluate how good it is to be in a given situation. There are two closely related types.

- The state-value function evaluates a state \(s\) under policy \(\pi\):

$$
v_\pi(s) = \mathbb{E}_\pi \big[ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \mid S_t = s \big],
$$

where \(\gamma \in [0,1]\) is the discount factor.
- The action-value function (also called the \(Q\)-function) evaluates a state--action pair. It answers: if I am in state \(s\), take action \(a\) now, and then follow policy \(\pi\), what long-term reward should I expect?

$$
q_\pi(s,a) = \mathbb{E}_\pi \big[ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \mid S_t = s,\; A_t = a \big].
$$

Action-value functions are especially useful for action selection, since they directly compare which action looks better in the same state.

### Model

A model captures how the environment behaves. While a policy tells the agent what action to take and a value function predicts long-term reward, a model predicts what the environment will do next. Having a model allows the agent to reason about the consequences of actions without directly interacting with the real environment.

In the standard RL setting, a model consists of two parts.

- The transition model predicts the next state given the current state and action:

$$
\mathcal{P}^{a}_{ss'} = \mathbb{P}[S_{t+1} = s' \mid S_t = s,\; A_t = a].
$$

- The reward model predicts the expected immediate reward for taking an action in a state:

$$
\mathcal{R}^{a}_{s} = \mathbb{E}[R_{t+1} \mid S_t = s,\; A_t = a].
$$

Together, these components describe the dynamics of the environment as seen by the agent. Algorithms that explicitly learn or use \(\mathcal{P}\) and \(\mathcal{R}\) are called model-based methods. In contrast, model-free methods do not build an explicit model of the environment. They learn a policy and/or value functions directly from experience. Most of our discussion will focus on model-free RL, since it is often simpler to implement and is widely used in practice.

## Categorizing RL Agents

### Classification of RL Agents by Learned Components

RL algorithms can be categorized by what they explicitly learn: a value function, a policy, or both. This classification helps clarify the differences between methods and when each is most appropriate.

- Value-based agents focus on learning a value function, usually an action-value function \(q(s,a)\). They do not explicitly learn a policy. Instead, the policy is implicit: at each time step, actions are chosen by comparing values and picking the best one (for example, choosing the action with the highest \(q(s,a)\)). Classic examples like Q-learning fall into this category.
- Policy-based agents directly learn a policy \(\pi(a \mid s)\) without maintaining a value function. The policy is optimized to maximize expected cumulative reward, often using gradient-based methods. These approaches are natural for continuous action spaces and stochastic policies, but they do not explicitly evaluate how good states or actions are.
- Actor-critic agents combine both ideas. The actor is the policy, which selects actions, while the critic is a value function that evaluates how good those actions are. The critic provides feedback to improve the actor, making learning more stable and efficient. Many modern RL algorithms use this structure because it balances the strengths of value-based and policy-based methods.

### Categorizing RL Agents: Model-Free vs. Model-Based

Another common way to categorize RL agents is based on whether they use an explicit model of the environment. As seen in the previous sectoin we can define these as:

- Model-free agents do not try to learn how the environment works internally. Instead, they learn a policy, a value function, or both, directly from experience. All learning happens through trial and error, using observed rewards and transitions, without explicitly predicting the next state or reward. Most standard RL algorithms fall into this category, and most of this discussion focuses on model-free methods because they are simple, flexible, and widely used in practice.
- Model-based agents, on the other hand, explicitly learn or are given a model of the environment. In addition to learning a policy and/or value function, they also learn how states transition and what rewards to expect. This allows the agent to plan ahead by simulating future outcomes before acting. Model-based methods can be more data-efficient, but they are often harder to design and computationally more expensive.

This distinction also clarifies the difference between learning and planning in sequential decision making: both aim to improve the agent's policy, but they differ in what information is available. In model-free reinforcement learning, the environment is initially unknown, so the agent must learn through interaction and trial and error. In planning (model-based), a model of the environment is known or already learned, allowing the agent to simulate future trajectories and improve its policy internally through deliberation, reasoning, or search.

## Exploration and Exploitation

Reinforcement learning is often described as trial-and-error learning. The agent must discover a good policy by interacting with the environment and learning from the outcomes of its actions. At the same time, it wants to collect as much reward as possible while learning, rather than performing poorly for a long time.

This leads to a fundamental trade-off. Exploration refers to taking actions that may not seem optimal right now, but help the agent gather more information about the environment. Exploitation, on the other hand, means using the information the agent already has to choose actions that are expected to give high reward.

Both are necessary. If the agent only exploits, it may get stuck with a suboptimal policy because it never tries alternatives. If it only explores, it may learn a lot but fail to accumulate reward. Effective RL algorithms balance exploration and exploitation so that the agent learns about the environment while still performing reasonably well along the way.

### Prediction and Control

Many problems in reinforcement learning can be viewed through the lens of prediction and control. These are two closely related, but distinct, objectives.

- Prediction asks: given a fixed policy, how good is it? The goal is to evaluate the future by estimating expected rewards when the agent follows a particular policy. This is typically done using value functions, which predict long-term reward without changing the policy itself.
- Control goes one step further. Instead of just evaluating a policy, the goal is to improve it. Control is about optimizing the future by finding the best possible policy, one that maximizes expected cumulative reward. Most RL algorithms alternate between prediction (evaluating how good things are) and control (using that information to choose better actions).
