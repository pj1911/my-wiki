# Fuzzy Inference Systems

## Introduction
Fuzzy inference systems sound mysterious if we are hearing about “fuzzy” things for the first time. In reality, they are just a way to let computers handle vague, human-style reasoning like “temperature is warm” or “speed is a bit high”. This chapter builds up to fuzzy inference systems, starting with classical sets and logic, then introducing fuzzy sets, and finally showing how fuzzy rules come together into a full fuzzy inference system.

## Why care about fuzziness?

Let us start with an everyday question: Is this coffee hot? A typical answer is not just “yes” or “no”. we might say:

- “It's lukewarm” or “It's kind of hot, but we can drink it.”

Human language is full of words like *tall*, *young*, *expensive*, *close*, *fast*, etc. These are not crisp categories with sharp boundaries. Classical logic and classical sets, however, are based on sharp yes/no decisions: either \(x\) is in the set \(A\), or it is not. Fuzzy logic and fuzzy inference systems were developed to bridge this gap: they allow us to mathematically model concepts like “kind of hot” or “very tall” and use them in reasoning and control. Before we get to fuzzy stuff, let us quickly recall how classical sets and logic work, and why they struggle with vagueness.

## Classical sets and logic (crisp world)

### Crisp sets

Consider some collection of objects we care about, called the *universe of discourse* and usually denoted by \(X\).
For example:

$$
X = \{\text{all real temperatures between } 0^\circ\mathrm{C} \text{ and } 100^\circ\mathrm{C}\}.
$$

A *crisp set* \(A\) is a subset of \(X\). Membership is all-or-nothing: an element \(x \in X\) either belongs to \(A\) or it does not. We can express this with an *indicator function* (also called a characteristic function):

$$
\chi_A(x) =
\begin{cases}
1 & \text{if } x \in A,\
0 & \text{if } x \notin A.
\end{cases}
$$

For example, let \(X\) be temperatures and define the set

$$
A = \{\text{temperatures considered “hot”}\}.
$$

In classical set theory, we must draw a sharp line: maybe \(\ A = \{ x \in X : x \ge 60^\circ\mathrm{C} \}\). Then:

$$
\chi_A(59.9^\circ\mathrm{C}) = 0, \quad \chi_A(60^\circ\mathrm{C}) = 1.
$$

### Classical logic

Classical logic deals with *propositions* that are either true or false. We use operations like: NOT (\(\neg\)), AND (\(\land\)), OR (\(\lor\)).
If \(P\) and \(Q\) are true/false statements, then:

- \(\neg P\) is true when \(P\) is false.
- \(P \land Q\) is true when both \(P\) and \(Q\) are true.
- \(P \lor Q\) is true when at least one of them is true.

This matches the crisp set view nicely: being in a set is a boolean property, and combining sets (intersection, union, complement) mirrors these logical operations.

### Where crisp logic struggles

The problem shows up with vague concepts: “Is Alice tall?”. If we are forced to answer strictly yes/no, we:

- either draw an arbitrary boundary (e.g., “tall if height \(\ge 170\)cm”), or
- end up disagreeing, because different people choose different boundaries.

There is no natural, universally accepted sharp cut-off where “not tall” suddenly becomes “tall”. The concept itself is inherently *gradual*. This is where fuzzy sets and fuzzy logic come in.

## Vagueness vs randomness

Before introducing fuzziness, it is useful to distinguish it from *randomness*. These are different kinds of uncertainty. Probability theory deals with randomness or lack of knowledge about which outcome will occur. For example, we toss a fair coin. We do not know if it will land heads or tails, but after the toss it is definitely either heads or tails. A probability \(P(\text{heads}) = 0.5\) does *not* mean the coin is “half-heads and half-tails”. It means, roughly, that in the long run we will see heads about half the time.

### Vagueness (fuzziness)

Consider the question: "Is a \(168\) cm person tall?". There is no randomness here. We are not uncertain about the height, we already know it exactly. The uncertainty is about how well this precise height fits a vague category like “tall”. Instead of forcing a yes/no answer, fuzzy logic lets us assign a number that says how well something fits a vague description. For instance, we might say:

$$
\text{``Tall''}(168\text{ cm}) = 0.6,
$$

meaning “168 cm is tall to degree \(0.6\)”. This is a different numerical quantity than a probability. With that distinction in mind, we can now define fuzzy sets.

## Fuzzy sets: gradual membership

### From indicator functions to membership functions

Recall, if \(X\) is a set defined on temperatures and A is a subset of it, defined as

$$
A = \{\text{temperatures considered ``hot''}\}.
$$

In classical set theory, we must draw a sharp line: maybe \(A = \{ x \in X : x \ge 60^\circ\mathrm{C} \}\). We can express this with an *indicator or membership function*:

$$
\chi_A(x) =
\begin{cases}
1 & \text{if } x \in A,\
0 & \text{if } x \notin A.
\end{cases}
$$

But, a *fuzzy set* \(A\) on \(X\) is defined by a *membership function*:

$$
\mu_A : X \to [0,1].
$$

For each element \(x \in X\), the value \(\mu_A(x)\) tells us the *degree* to which \(x\) belongs to the fuzzy set \(A\):

- \(\mu_A(x) = 0\) means “definitely not in \(A\)”,
- \(\mu_A(x) = 1\) means “fully in \(A\)”,
- values between \(0\) and \(1\) represent partial membership.

### Basic fuzzy set operations

Given fuzzy sets \(A\) and \(B\) with membership functions \(\mu_A(x)\) and \(\mu_B(x)\), we can define operations analogous to union, intersection, and complement. One common choice is:

**Complement:**

$$
\mu_{\neg A}(x) = 1 - \mu_A(x).
$$

**Intersection (AND):**

$$
\mu_{A \cap B}(x) = \min\big(\mu_A(x), \mu_B(x)\big).
$$

**Union (OR):**

$$
\mu_{A \cup B}(x) = \max\big(\mu_A(x), \mu_B(x)\big).
$$

These are not the only possible definitions, but they are intuitive and widely used:

- For “\(x\) is in both \(A\) and \(B\)”, the degree is limited by the smaller membership.
- For “\(x\) is in \(A\) or \(B\)”, the degree is given by the larger membership.

So far, we have only talked about static fuzzy sets. To build fuzzy inference systems, we need to connect fuzzy sets with variables and rules.

## Linguistic variables and fuzzy rules

### Linguistic variables

A *linguistic variable* is a variable whose values are words (or short phrases) rather than numbers. For example, instead of writing `Temperature = 25°C`, we might describe it as

$$
\texttt{Temperature} \in \{\text{'cold'}, \text{'cool'}, \text{'warm'}, \text{'hot'}\}.
$$

Each word here is not just a label, it corresponds to a fuzzy set on the underlying numeric domain. Concretely:

- the *universe of discourse* is the numeric range of possible temperatures (e.g. \(0\)–\(100^\circ\mathrm{C}\)),
- each linguistic term (“cold”, “cool”, "warm", "hot", etc.) is represented by its own fuzzy set over this range.

For example, “cold” might be modeled so that

$$
\mu_{\text{cold}}(10^\circ\mathrm{C}) = 1,\quad
\mu_{\text{cold}}(15^\circ\mathrm{C}) = 0.5,\quad
\mu_{\text{cold}}(25^\circ\mathrm{C}) = 0,
$$

which means \(10^\circ\mathrm{C}\) is fully cold, \(15^\circ\mathrm{C}\) is “half cold”, and \(25^\circ\mathrm{C}\) is not cold at all.

Thus a single numeric value, such as \(25^\circ\mathrm{C}\), can belong to several of these fuzzy sets at once, but with different degrees of membership.

### Fuzzy IF–THEN rules

Now we can express knowledge in a very natural form using fuzzy sets. For instance, in a simple temperature control scenario, we might want to decide how fast a fan should run based on how warm the room feels. A human would say something like: **IF** temperature is warm **THEN** fan speed is medium. More formally, this is a fuzzy IF–THEN rule of the form:

$$
\text{IF } x \text{ is } A \text{ THEN } y \text{ is } B.
$$

Here:

- \(x\) is an input variable (e.g., temperature),
- \(A\) is a fuzzy set on the \(x\)-domain (e.g., “warm”),
- \(y\) is an output variable (e.g., fan speed),
- \(B\) is a fuzzy set on the \(y\)-domain (e.g., “medium”).

A fuzzy inference system will take a collection of such rules, combine them, and produce an output given specific inputs. To understand how, we need to look at the structure of a fuzzy inference system.

## What is a fuzzy inference system?

A *fuzzy inference system* (FIS) is a mapping from inputs to outputs using fuzzy logic. It usually has the following components:

1. **Fuzzification**: convert numerical inputs into degrees of membership in fuzzy sets.
2. **Rule evaluation (inference)**: compute how strongly each fuzzy rule is activated by the current inputs.
3. **Aggregation**: combine the effects of all rules to obtain fuzzy output sets.
4. **Defuzzification**: convert the fuzzy output sets back into crisp numerical outputs.

We now walk through one concrete fuzzy inference system from start to finish.

**Setup.**  
We use two inputs and one output:

- Temperature \(T\) in \([0,40]\) (\(^\circ\mathrm{C}\)).
- Humidity \(H\) in \([0,100]\) (percent).
- Fan speed \(S\) in \([0,10]\) (arbitrary units).

For \(T\) we use two fuzzy sets:

$$
\text{'cool'}, \quad \text{'warm'},
$$

for \(H\) we use two:

$$
\text{'dry'}, \quad \text{'humid'},
$$

and for \(S\) we use three:

$$
\text{'low'}, \quad \text{'medium'}, \quad \text{'high'}.
$$

We will use the fixed input

$$
T = 24^\circ\mathrm{C}, \qquad H = 55\%.
$$

**Membership functions.**

**Temperature.**

$$
\mu_{\text{cool}}(T) =
\begin{cases}
1, & T \le 18,\
\dfrac{28 - T}{10}, & 18 < T < 28,\
0, & T \ge 28,
\end{cases}
\qquad
\mu_{\text{warm}}(T) =
\begin{cases}
0, & T \le 20,\
\dfrac{T - 20}{10}, & 20 < T < 30,\
1, & T \ge 30.
\end{cases}
$$

**Humidity.**

$$
\mu_{\text{dry}}(H) =
\begin{cases}
1, & H \le 40,\
\dfrac{60 - H}{20}, & 40 < H < 60,\
0, & H \ge 60,
\end{cases}
\qquad
\mu_{\text{humid}}(H) =
\begin{cases}
0, & H \le 50,\
\dfrac{H - 50}{30}, & 50 < H < 80,\
1, & H \ge 80.
\end{cases}
$$

**Fan speed.** (triangular shapes)

$$
\mu_{\text{low}}(S) =
\begin{cases}
\dfrac{S}{2}, & 0 < S \le 2,\
\dfrac{4 - S}{2}, & 2 < S < 4,\
0, & \text{otherwise},
\end{cases}
$$

$$
\mu_{\text{medium}}(S) =
\begin{cases}
\dfrac{S - 3}{2}, & 3 < S \le 5,\
\dfrac{7 - S}{2}, & 5 < S < 7,\
0, & \text{otherwise},
\end{cases}
$$

$$
\mu_{\text{high}}(S) =
\begin{cases}
\dfrac{S - 6}{2}, & 6 < S \le 8,\
\dfrac{10 - S}{2}, & 8 < S < 10,\
0, & \text{otherwise}.
\end{cases}
$$

### Step 1: Fuzzification

For \(T = 24\):

$$
\mu_{\text{cool}}(24) = \frac{28 - 24}{10} = 0.4, \qquad
\mu_{\text{warm}}(24) = \frac{24 - 20}{10} = 0.4.
$$

For \(H = 55\):

$$
\mu_{\text{dry}}(55) = \frac{60 - 55}{20} = 0.25, \qquad
\mu_{\text{humid}}(55) = \frac{55 - 50}{30} \approx 0.17.
$$

So the crisp input \((T,H) = (24,55)\) becomes the fuzzy description

$$
\mu_{\text{cool}}(24) = 0.4, \quad
\mu_{\text{warm}}(24) = 0.4, \quad
\mu_{\text{dry}}(55) = 0.25, \quad
\mu_{\text{humid}}(55) \approx 0.17.
$$

### Step 2: Rule evaluation

We use three rules:

- **R1:** IF \(T\) is cool AND \(H\) is dry THEN \(S\) is low.
- **R2:** IF \(T\) is warm AND \(H\) is dry THEN \(S\) is medium.
- **R3:** IF \(T\) is warm AND \(H\) is humid THEN \(S\) is high.

We take AND as minimum. The firing strengths \(\alpha_k\) are:

$$
\alpha_1 = \min(0.4, 0.25) = 0.25,\\
\alpha_2 = \min(0.4, 0.25) = 0.25,\\
\alpha_3 = \min(0.4, 0.17) \approx 0.17.
$$

So all three rules fire, with different strengths.

### Step 3: Aggregation of rule outputs

Each rule \(R_k\) has:

- a firing strength \(\alpha_k\),
- an output fuzzy set (\(\mu_{\text{low}}, \mu_{\text{medium}}, \mu_{\text{high}}\)).

**Inside a single rule**

Example: R2

$$
\mu_{S|\mathrm{R2}}(S) = \min\big(\alpha_2, \mu_{\text{medium}}(S)\big).
$$

If

$$
\mu_{\text{medium}}(4.5) = 0.8,\qquad
\mu_{\text{medium}}(3.0) = 0.2,
$$

then

$$
\mu_{S|\mathrm{R2}}(4.5) = 0.25,\qquad
\mu_{S|\mathrm{R2}}(3.0) = 0.2.
$$

We do this for all rules:

$$
\mu_{S|\mathrm{R1}}(S) = \min(\alpha_1, \mu_{\text{low}}(S)),\\
\mu_{S|\mathrm{R2}}(S) = \min(\alpha_2, \mu_{\text{medium}}(S)),\\
\mu_{S|\mathrm{R3}}(S) = \min(\alpha_3, \mu_{\text{high}}(S)).
$$

**Across rules**

For some \(S^\star\):

$$
\mu_{\text{out}}(S^\star)
= \max\big(\mu_{S|\mathrm{R1}}(S^\star), \mu_{S|\mathrm{R2}}(S^\star), \mu_{S|\mathrm{R3}}(S^\star)\big).
$$

This gives the final aggregated output fuzzy set:

$$
\mu_{\text{out}}(S)
= \max\big(\mu_{S|\mathrm{R1}}(S), \mu_{S|\mathrm{R2}}(S), \mu_{S|\mathrm{R3}}(S)\big).
$$

**Why min–max?**  
Mamdani’s scheme is intuitive, matches crisp logic when membership is 0/1, easy to visualize, and easy to implement.

Other options include product inference, soft OR aggregators, or Sugeno-type consequents:

$$
S^\ast = \frac{\sum_k \alpha_k f_k(T,H)}{\sum_k \alpha_k}.
$$

### Step 4: Defuzzification

Using the centroid:

$$
S^\ast =
\frac{\displaystyle \int_0^{10} S \, \mu_{\text{out}}(S)\, dS}
     {\displaystyle \int_0^{10}     \mu_{\text{out}}(S)\, dS}.
$$

For our shapes:

$$
S^\ast \approx 4.7.
$$

Given \(T = 24^\circ\mathrm{C}\) and \(H = 55\%\), the fuzzy inference system suggests a fan speed slightly below the center of “medium”.

## When are fuzzy inference systems useful?

FIS are useful when:

- human expertise can be expressed as fuzzy rules,
- systems are too complex or poorly understood for precise models,
- linguistic terms describe inputs/outputs well.

Common in:

- consumer products,
- control and automation,
- decision-making systems.

### Limitations

- Designing good membership functions and rules is domain-specific.
- They do not learn unless combined with machine learning.
- Rule bases can blow up in high dimensions.

## Wrapping up

We started from classical sets and logic, which struggle with vague concepts. Fuzzy sets extend membership to degrees in \([0,1]\). Using fuzzy sets, we defined linguistic variables and fuzzy rules. A fuzzy inference system:

1. fuzzifies inputs,
2. evaluates fuzzy rules,
3. aggregates their outputs,
4. defuzzifies to produce a crisp output.

The result is a system that can reason in a way that resembles human language.
