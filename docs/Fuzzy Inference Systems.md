# Fuzzy Inference Systems

## Introduction

Fuzzy inference systems sound mysterious if we are hearing about 'fuzzy' things for the first time. In reality, they are just a way to let computers handle vague, human-style reasoning like 'temperature is warm' or 'speed is a bit high'. This chapter builds up to fuzzy inference systems, starting with classical sets and logic, then introducing fuzzy sets, and finally showing how fuzzy rules come together into a full fuzzy inference system.

## Why care about fuzziness?

Let us start with an everyday question: Is this coffee hot? A typical answer is not just ''yes'' or ''no''. we might say:

- ''It's lukewarm'' or ''It's kind of hot, but we can drink it.''

Human language is full of words like *tall*, *young*, *expensive*, *close*, *fast*, etc. These are not crisp categories with sharp boundaries. Classical logic and classical sets, however, are based on sharp yes/no decisions: either \(x\) is in the set \(A\), or it is not. Fuzzy logic and fuzzy inference systems were developed to bridge this gap: they allow us to mathematically model concepts like 'kind of hot' or 'very tall' and use them in reasoning and control. Before we get to fuzzy stuff, let us quickly recall how classical sets and logic work, and why they struggle with vagueness.

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
1 & \text{if } x \in A,\\
0 & \text{if } x \notin A.
\end{cases}
$$

For example, let \(X\) be temperatures and define the set

$$
A = \{\text{temperatures considered 'hot'}\}.
$$

In classical set theory, we must draw a sharp line: maybe \(A = \{ x \in X : x \ge 60^\circ\mathrm{C} \}\). Then:

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

The problem shows up with vague concepts: 'Is Alice tall?'. If we are forced to answer strictly yes/no, we:

- either draw an arbitrary boundary (e.g., 'tall if height \(\ge 170\)cm'), or
- end up disagreeing, because different people choose different boundaries.

There is no natural, universally accepted sharp cut-off where 'not tall' suddenly becomes 'tall'. The concept itself is inherently *gradual*. This is where fuzzy sets and fuzzy logic come in.

## Vagueness vs randomness

Before introducing fuzziness, it is useful to distinguish it from *randomness*. These are different kinds of uncertainty. Probability theory deals with randomness or lack of knowledge about which outcome will occur. For example, we toss a fair coin. We do not know if it will land heads or tails, but after the toss it is definitely either heads or tails. A probability \(P(\text{heads}) = 0.5\) does *not* mean the coin is 'half-heads and half-tails'. It means, roughly, that in the long run we will see heads about half the time.

### Vagueness (fuzziness)

Consider the question: "Is a \(168\) cm person tall?". There is no randomness here. We are not uncertain about the height, we already know it exactly. The uncertainty is about how well this precise height fits a vague category like 'tall'. Instead of forcing a yes/no answer, fuzzy logic lets us assign a number that says how well something fits a vague description.
 For instance, we might say:

$$
\text{"Tall"}(168\text{ cm}) = 0.6,
$$

meaning '168 cm is tall to degree \(0.6\)'. This is a different numerical quantity than a probability. With that distinction in mind, we can now define fuzzy sets.

## Fuzzy sets: gradual membership

### From indicator functions to membership functions

Recall, if \(X\) is a set defined on temperatures and A is a subset of it, defined as

$$
A = \{\text{temperatures considered "hot"}\}.
$$

In classical set theory, we must draw a sharp line: maybe \(A = \{ x \in X : x \ge 60^\circ\mathrm{C} \}\). We can express this with an *indicator or membership function*:

$$
\chi_A(x) =
\begin{cases}
1 & \text{if } x \in A,\\
0 & \text{if } x \notin A.
\end{cases}
$$

But, a *fuzzy set* \(A\) on \(X\) is defined by a *membership function*:

$$
\mu_A : X \to [0,1].
$$

For each element \(x \in X\), the value \(\mu_A(x)\) tells us the *degree* to which \(x\) belongs to the fuzzy set \(A\):

- \(\mu_A(x) = 0\) means 'definitely not in \(A\)',
- \(\mu_A(x) = 1\) means 'fully in \(A\)',
- values between \(0\) and \(1\) represent partial membership.

### Basic fuzzy set operations

Given fuzzy sets \(A\) and \(B\) with membership functions \(\mu_A(x)\) and \(\mu_B(x)\), we can define operations analogous to union, intersection, and complement. One common choice is:

- **Complement:**

$$
\mu_{\neg A}(x) = 1 - \mu_A(x).
$$

- **Intersection (AND):**

$$
\mu_{A \cap B}(x) = \min\big(\mu_A(x), \mu_B(x)\big).
$$

- **Union (OR):**

$$
\mu_{A \cup B}(x) = \max\big(\mu_A(x), \mu_B(x)\big).
$$

These are not the only possible definitions, but they are intuitive and widely used:

- For '\(x\) is in both \(A\) and \(B\)', the degree is limited by the smaller membership.
- For '\(x\) is in \(A\) or \(B\)', the degree is given by the larger membership.

So far, we have only talked about static fuzzy sets. To build fuzzy inference systems, we need to connect fuzzy sets with variables and rules.

## Linguistic variables and fuzzy rules

### Linguistic variables

A *linguistic variable* is a variable whose values are words (or short phrases) rather than numbers. For example, instead of writing Temperature \(= 25^\circ\mathrm{C}\), we might describe it as

$$
\texttt{Temperature} \in \{\text{'cold'}, \text{'cool'}, \text{'warm'}, \text{'hot'}\}.
$$

Each word here is not just a label, it corresponds to a fuzzy set on the underlying numeric domain. Concretely:

- the *universe of discourse* is the numeric range of possible temperatures (e.g. \(0\)--\(100^\circ\mathrm{C}\)),
- each linguistic term ('cold', 'cool', 'warm', 'hot', etc.) is represented by its own fuzzy set over this range.

For example, 'cold' might be modeled so that

$$
\mu_{\text{cold}}(10^\circ\mathrm{C}) = 1,\quad
\mu_{\text{cold}}(15^\circ\mathrm{C}) = 0.5,\quad
\mu_{\text{cold}}(25^\circ\mathrm{C}) = 0,
$$

which means \(10^\circ\)C is fully cold, \(15^\circ\)C is 'half cold', and \(25^\circ\)C is not cold at all.

Thus a single numeric value, such as \(25^\circ\mathrm{C}\), can belong to several of these fuzzy sets at once, but with different degrees of membership.

### Fuzzy IF-THEN rules

Now we can express knowledge in a very natural form using fuzzy sets. For instance, in a simple temperature control scenario, we might want to decide how fast a fan should run based on how warm the room feels. A human would say something like: **IF** temperature is warm **THEN** fan speed is medium. More formally, this is a fuzzy IF-THEN rule of the form:

$$
\text{IF } x \text{ is } A \text{ THEN } y \text{ is } B.
$$

Here:

- \(x\) is an input variable (e.g., temperature),
- \(A\) is a fuzzy set on the \(x\)-domain (e.g., 'warm'),
- \(y\) is an output variable (e.g., fan speed),
- \(B\) is a fuzzy set on the \(y\)-domain (e.g., 'medium').

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
\text{"cool"}, \quad \text{"warm"},
$$

for \(H\) we use two:

$$
\text{"dry"}, \quad \text{"humid"},
$$

and for \(S\) we use three:

$$
\text{"low"}, \quad \text{"medium"}, \quad \text{"high"}.
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
1, & T \le 18,\\
\dfrac{28 - T}{10}, & 18 < T < 28,\\
0, & T \ge 28,
\end{cases}
\qquad
\mu_{\text{warm}}(T) =
\begin{cases}
0, & T \le 20,\\
\dfrac{T - 20}{10}, & 20 < T < 30,\\
1, & T \ge 30.
\end{cases}
$$

**Humidity.**

$$
\mu_{\text{dry}}(H) =
\begin{cases}
1, & H \le 40,\\
\dfrac{60 - H}{20}, & 40 < H < 60,\\
0, & H \ge 60,
\end{cases}
\qquad
\mu_{\text{humid}}(H) =
\begin{cases}
0, & H \le 50,\\
\dfrac{H - 50}{30}, & 50 < H < 80,\\
1, & H \ge 80.
\end{cases}
$$

**Fan speed.** (triangular shapes)

$$
\begin{aligned}
\mu_{\text{low}}(S) &=
\begin{cases}
\dfrac{S}{2}, & 0 < S \le 2,\\
\dfrac{4 - S}{2}, & 2 < S < 4,\\
0, & \text{otherwise},
\end{cases}
\\
\mu_{\text{medium}}(S) &=
\begin{cases}
\dfrac{S - 3}{2}, & 3 < S \le 5,\\
\dfrac{7 - S}{2}, & 5 < S < 7,\\
0, & \text{otherwise},
\end{cases}
\\
\mu_{\text{high}}(S) &=
\begin{cases}
\dfrac{S - 6}{2}, & 6 < S \le 8,\\
\dfrac{10 - S}{2}, & 8 < S < 10,\\
0, & \text{otherwise}.
\end{cases}
\end{aligned}
$$

**Step 1: Fuzzification.**

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

**Step 2: Rule evaluation.**

We use three rules:

- **R1:** IF \(T\) is cool AND \(H\) is dry THEN \(S\) is low.
- **R2:** IF \(T\) is warm AND \(H\) is dry THEN \(S\) is medium.
- **R3:** IF \(T\) is warm AND \(H\) is humid THEN \(S\) is high.

We take AND as minimum (discussed above). The firing strength \(\alpha_k\) of rule \(k\) is then given by:

$$
\begin{aligned}
\alpha_1 &= \min\big(\mu_{\text{cool}}(24), \mu_{\text{dry}}(55)\big)
        = \min(0.4, 0.25) = 0.25,\\
\alpha_2 &= \min\big(\mu_{\text{warm}}(24), \mu_{\text{dry}}(55)\big)
        = \min(0.4, 0.25) = 0.25,\\
\alpha_3 &= \min\big(\mu_{\text{warm}}(24), \mu_{\text{humid}}(55)\big)
        = \min(0.4, 0.17) \approx 0.17.
\end{aligned}
$$

So all three rules fire, with different strengths.

**Step 3: Aggregation of rule outputs.**

By this point, each rule \(R_k\) has:

- a firing strength \(\alpha_k \in [0,1]\) that says how true its IF-part is for the current input, and
- an output fuzzy set (low, medium and high) that tells, for each possible fan speed \(S \in [0,10]\), how well that fan speed fits the label in the THEN-part if the rule were fully true (\(\mu_{\text{low}}\), \(\mu_{\text{medium}}\) and \(\mu_{\text{high}}\)).

Step 3 answers the question: *Given the firing strengths \(\alpha_k\) of all rules and their corresponding output fuzzy sets (such as \(\mu_{\text{low}}\), \(\mu_{\text{medium}}\), and \(\mu_{\text{high}}\) over fan speeds \(S\)), how strongly does the rule base as a whole support each possible fan speed \(S\)?* We do this in two stages:

To keep things concrete, suppose from Step 2 we already have

$$
\alpha_1 = 0.25, \qquad
\alpha_2 = 0.25, \qquad
\alpha_3 = 0.17.
$$

**Inside a single rule:**

Take one rule in isolation, for example

$$
\text{R2: IF } T \text{ is warm AND } H \text{ is dry THEN } S \text{ is medium.}
$$

For the current input \((T,H)\), the IF-part (the antecedent) is not fully true, its truth is \(\alpha_2 = 0.25\). The fuzzy set \(\mu_{\text{medium}}(S)\) tells us, for each \(S\), how 'medium' that speed would be *if the rule were completely true*. But in reality the rule is only true to degree \(0.25\), so its recommendation should be weakened accordingly. For each \(S\), we therefore combine:

- the degree to which \(S\) is 'medium': \(\mu_{\text{medium}}(S)\),
- the degree to which the rule is true: \(\alpha_2 = 0.25\).

The standard Mamdani choice (a classic and widely used fuzzy inference scheme, chosen here because it is simple, interpretable, and closely follows human IF-THEN rules) is

$$
\mu_{S|\mathrm{R2}}(S) = \min\big(\alpha_2, \mu_{\text{medium}}(S)\big).
$$

For example, imagine that from the output membership function we know:

$$
\mu_{\text{medium}}(4.5) = 0.8, \qquad
\mu_{\text{medium}}(3.0) = 0.2.
$$

Then R2's support for these two speeds becomes

$$
\mu_{S|\mathrm{R2}}(4.5) = \min(0.25, 0.8) = 0.25,\qquad
\mu_{S|\mathrm{R2}}(3.0) = \min(0.25, 0.2) = 0.2.
$$

So:

- At \(S=4.5\), the rule would like this speed quite a lot ('medium' to degree \(0.8\)), but the rule itself is weak (\(0.25\)), so we cap the support at \(0.25\).
- At \(S=3.0\), the rule is equally weak (\(0.25\)) but \(S=3.0\) is barely 'medium' (\(0.2\)), so the support is only \(0.2\).

In other words, the rule's support for a particular \(S\) is always the weaker of:

\[
    "\text{how true is the rule?}" \quad \text{vs.} \quad "\text{how well does } S \text{ fit the label in the THEN-part?}"
\]

This minimum operator also has two nice properties:

- If everything is crisp (\(\alpha_2 \in \{0,1\}\) and \(\mu_{\text{medium}}(S) \in \{0,1\}\)), this reduces to ordinary logic: the consequent is true only when both the rule fires and \(S\) satisfies the label.
- Geometrically, it corresponds to 'cutting' the top of the fuzzy set \(\mu_{\text{medium}}(S)\) at height \(\alpha_2\). When plotted, each rule produces a 'truncated hill' over the output axis.

We do the same thing for every rule, so that each fuzzy IF-THEN rule contributes its own truncated output membership function, i.e. its own partial, weighted opinion about \(S\). Formally, in our example:

$$
\begin{aligned}
\mu_{S|\mathrm{R1}}(S) &= \min\big(\alpha_1, \mu_{\text{low}}(S)\big),\\
\mu_{S|\mathrm{R2}}(S) &= \min\big(\alpha_2, \mu_{\text{medium}}(S)\big),\\
\mu_{S|\mathrm{R3}}(S) &= \min\big(\alpha_3, \mu_{\text{high}}(S)\big).
\end{aligned}
$$

**Across rules:**

Now we fix a particular output value \(S^\star\) (say \(S^\star=4.5\)). Suppose the truncated fuzzy sets from the three rules give:

$$
\mu_{S|\mathrm{R1}}(4.5) = 0.10, \qquad
\mu_{S|\mathrm{R2}}(4.5) = 0.25, \qquad
\mu_{S|\mathrm{R3}}(4.5) = 0.05.
$$

This means:

- Rule 1 says: '\(S^\star\) is low' to degree \(0.10\).
- Rule 2 says: '\(S^\star\) is medium' to degree \(0.25\).
- Rule 3 says: '\(S^\star\) is high' to degree \(0.05\).

The question is: *overall, how acceptable is \(S^\star\) as an output according to the whole rule base?* The intuitive answer is: \(S^\star\) is acceptable if at least one rule supports it, and the more strongly any rule supports it, the more acceptable it is. This is a fuzzy version of logical OR over rules. In standard fuzzy logic, OR is modeled by the maximum of the truth degrees, so we take:

$$
\mu_{\text{out}}(S^\star)
= \max\big(\mu_{S|\mathrm{R1}}(S^\star), \mu_{S|\mathrm{R2}}(S^\star), \mu_{S|\mathrm{R3}}(S^\star)\big)
= \max(0.10, 0.25, 0.05) = 0.25.
$$

Doing this for every \(S\) gives the final aggregated output fuzzy set:

$$
\mu_{\text{out}}(S)
= \max\big(\mu_{S|\mathrm{R1}}(S), \mu_{S|\mathrm{R2}}(S), \mu_{S|\mathrm{R3}}(S)\big).
$$

Geometrically, this means we place all the truncated hills from the individual rules on the same axis and, at each \(S\), take the highest one. The resulting outline is the combined fuzzy belief about \(S\) after considering all rules.

**Why this min-max scheme, and what else is possible?**

The min-inside / max-across pattern is the classic Mamdani choice because:

- it is easy to interpret: each rule's influence is limited by its truth, and the system accepts an output if any rule supports it.
- it matches crisp logic when all memberships are \(0\) or \(1\).
- it is very easy to visualize and implement.

However, it is not the only design:

- Instead of \(\min(\alpha_k, \mu(S))\) inside a rule, one can use product \(\alpha_k \cdot \mu(S)\). For example, if \(\alpha_2 = 0.25\) and \(\mu_{\text{medium}}(4.5) = 0.8\), product inference would give \(0.25 \cdot 0.8 = 0.20\) instead of \(\min(0.25,0.8)=0.25\), smoothly scaling the fuzzy set rather than flat-cutting it.
- Instead of combining rules with \(\max\), one can use other aggregation operators that behave like a soft OR and give slightly different shapes to \(\mu_{\text{out}}(S)\).
- We also have another type of system known as Sugeno-type systems, the consequents are not fuzzy sets at all but crisp functions of the inputs. For example, each rule might have

$$
\text{R}_k: \ \text{IF (conditions)} \ \text{THEN } S = f_k(T,H) = a_k T + b_k H + c_k.
$$

  After computing the firing strengths \(\alpha_k\) of all rules, the final output is a weighted average of these rule outputs:

$$
S^\ast = \frac{\sum_k \alpha_k \, f_k(T,H)}{\sum_k \alpha_k}.
$$

  So if two rules fire with \(\alpha_1 = 0.6\), \(\alpha_2 = 0.4\) and give \(f_1(T,H)=5\), \(f_2(T,H)=8\), then

$$
S^\ast = \frac{0.6 \cdot 5 + 0.4 \cdot 8}{0.6 + 0.4} = 6.2.
$$

We stick to the Mamdani min-max style here because it gives a clean first mental model: each rule draws a fuzzy bump over the output axis, cut at the level of its truth, and the system as a whole takes the upper envelope of all these bumps as its final fuzzy recommendation.

**Step 4: Defuzzification.**

To obtain a single crisp fan speed, we use the centroid (center of gravity) of \(\mu_{\text{out}}\):

$$
S^\ast =
\frac{\displaystyle \int_0^{10} S \, \mu_{\text{out}}(S)\, dS}
     {\displaystyle \int_0^{10}     \mu_{\text{out}}(S)\, dS}.
$$

For the specific shapes and firing strengths above, this centroid is numerically around

$$
S^\ast \approx 4.7.
$$

So, given \(T = 24^\circ\mathrm{C}\) and \(H = 55\%\), the fuzzy inference system suggests a fan speed a bit below the center of 'medium'. Each step (fuzzification, rule evaluation, aggregation, defuzzification) is just an explicit, numerical way of turning vague rules into a concrete control action.

## When are fuzzy inference systems useful?

Fuzzy inference systems shine in situations where:

- Human expertise is available in the form of fuzzy rules like 'IF \(A\) is high AND \(B\) is low THEN ...'.
- The system is too complex or poorly understood to model precisely with differential equations or detailed physics.
- Inputs and outputs can be reasonably described using linguistic terms (e.g., control systems, decision support, heuristic policies).

They are common in:

- Consumer products (e.g., washing machines, cameras, air conditioners).
- Control and automation.
- Decision-making and ranking systems.

### Limitations

Fuzzy inference systems also have limitations:

- Designing good membership functions and rules can be somewhat artful and domain-specific.
- They do not magically 'learn' from data unless combined with learning methods (e.g., neuro-fuzzy systems).
- For very high-dimensional problems with many inputs, the rule base can explode combinatorially.

## Wrapping up

We started from classical sets and logic, where membership and truth are crisp, and saw that they struggle with inherently vague concepts like 'tall' or 'warm'. Fuzzy sets extend the idea of set membership to degrees in \([0,1]\), which can capture how well a specific value fits a vague category. Using fuzzy sets, we defined linguistic variables and fuzzy IF-THEN rules. A fuzzy inference system then:

1. fuzzifies numerical inputs,
2. evaluates fuzzy rules,
3. aggregates their fuzzy outputs,
4. and defuzzifies the aggregated result to get a crisp output.

The result is a system that can reason in a way that resembles human use of language.
