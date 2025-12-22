## Introduction
A first-order PDE is an equation for an unknown function $u$ of several variables where the highest derivatives that appear are only first derivatives. For $u=u(x,t)$, the most general way to write such an equation is

$$
F\!\big(x,t,\,u(x,t),\,u_x(x,t),\,u_t(x,t)\big)=0,
$$

meaning: at each point $(x,t)$, if we substitute the numbers $x,t,u,u_x,u_t$ into the function $F$, the result must be zero. This is called writing the PDE implicitly, as in this case the equation is not arranged to isolate any derivatives, it is just a constraint relating the variables and derivatives. Sometimes, we can rearrange the same PDE into an explicit form where one derivative is isolated, for example

$$
u_t = G(x,t,u,u_x)
\qquad\text{or equivalently}\qquad
u_t + a(x,t,u)\,u_x = b(x,t,u),
$$

but this rearrangement is not always possible globally (or it may become messy or ambiguous), so the implicit form is the most general starting point. 

### Implicit function theorem

But an implicit first-order PDE cannot always be rewritten in the explicit form. For example, say we have the implicit form

$$
F(x,t,u,u_x,u_t)=0.
$$

For many initial-value problems we would like to isolate the time derivative and write an ``evolution law''

$$
u_t = G(x,t,u,u_x),
$$

meaning: once $x,t,u,u_x$ are given, the equation selects one value of $u_t$ (at least locally). The key tool for deciding when an implicit PDE can be solved locally for a chosen derivative (or can be written explicitly) is the
Implicit Function Theorem (IFT) (we do not prove it here). But, for a general understanding, let $F$ be smooth and suppose at $(x_0,t_0)$ we have

$$
F(x_0,t_0,u_0,p_0,q_0)=0,
\qquad
u_0=u(x_0,t_0),\; p_0=u_x(x_0,t_0),\; q_0=u_t(x_0,t_0).
$$

If, in addition,

$$
\frac{\partial F}{\partial u_t}(x_0,t_0,u_0,p_0,q_0)\neq 0,
$$

then the IFT guarantees that near this point one can solve uniquely for $u_t$ as a smooth function of the other variables. This is plausible because if we freeze $(x,t,u,p)$ and view

$$
\phi(q)=F(x,t,u,p,q)
$$

as a one-variable function. At the base point, $\phi(q_0)=0$ (because $F(...)=0$) and $\phi'(q_0)\neq 0$, so $\phi$ crosses $0$ transversely and is locally monotone, hence the zero is isolated and unique. Small perturbations of $(x,t,u,p)$ change $\phi$ only slightly, so a nearby zero persist, the nonvanishing of $\partial F/\partial u_t$ keeps the crossing monotone, preserving uniqueness. Consequently, by IFT there is a smooth function $G$ such that, on some neighborhood of the base point $(x_0,t_0,u_0,p_0,q_0)$, the equation $F(x,t,u,p,q)=0$ can be solved uniquely for $q=u_t$ in terms of the other variables, i.e. $q=G(x,t,u,p)$. In other words, for $(x,t,u,u_x)$ sufficiently close to $(x_0,t_0,u_0,p_0)$, there exists exactly one $u_t$ near $q_0$ satisfying $F=0$, and this $u_t$ depends smoothly on $(x,t,u,u_x)$, so locally

$$
F(x,t,u,u_x,u_t)=0 \quad\Longleftrightarrow\quad u_t=G(x,t,u,u_x).
$$


If the condition fails, uniqueness (or even existence) can break. For example,

$$
u_t^2-u_x=0 \quad\Rightarrow\quad u_t=\pm\sqrt{u_x},
$$

so there are two branches for $u_x>0$, no real solutions for $u_x<0$, and branching occurs precisely where
$\frac{\partial F}{\partial u_t}=2u_t=0$.

## Initial value problem: Characteristics curves

Once a first-order PDE is in an explicit (or at least locally explicit) form, the most common problem type is an initial value problem (Cauchy problem): we provide the value of the unknown along an initial curve and ask the PDE to extend it away from that curve. For $u=u(x,t)$ a standard explicit model is the quasi-linear equation

$$
u_t + a(x,t,u)\,u_x = b(x,t,u),
$$

together with initial data at (say) $t=0$,

$$
u(x,0)=\phi(x).
$$

Now a simple question arises why is this not just a “plug and solve” type of equation? Because the PDE does not directly tell us $u$. It tells us how $u$ must change when we move in $(x,t)$ while keeping the relation between the partial slopes $u_t$ and $u_x$.

The key observation here is that the particular combination $u_t+a\,u_x$ has a very concrete meaning: it is the directional derivative of $u$ in the $(x,t)$-plane along a direction that tilts with slope $a$. To see this, pick any differentiable curve in the $(x,t)$-plane, written as $t\mapsto (x(t),t)$. Along this curve we can track how the value of the field changes by defining a single-variable function

$$
U(t):=u(x(t),t).
$$

By the ordinary chain rule,

$$
\frac{dU}{dt}=\frac{d}{dt}u(x(t),t)=u_x(x(t),t)\,x'(t)+u_t(x(t),t).
$$

So the rate of change of $u$ that an observer would measure while moving along the curve is

$$
\frac{d}{dt}u(x(t),t)=u_t+ x'(t)\,u_x.
$$

Now compare this with the PDE term $u_t+a\,u_x$: if we choose the curve so that its slope satisfies

$$
x'(t)=a(x(t),t,u(x(t),t)),
$$

then the observed change along that curve becomes exactly

$$
\frac{d}{dt}u(x(t),t)=u_t+a\,u_x.
$$

This is why, along those special curves, a PDE like $u_t+a\,u_x=b$ turns into the single ODE $\frac{d}{dt}u=b$.
 Indeed, along any differentiable curve $t\mapsto (x(t),t)$, if we choose $x'(t)=a(x,t,u)$, the PDE becomes the ODE

$$
\frac{d}{dt}u(x(t),t)=b(x,t,u).
$$

Thus the PDE can be attacked by finding special curves in the $(x,t)$-plane (later called characteristics) that start on the initial line $t=0$ and carry the initial values $\phi(x)$ forward by an ODE.

### From a PDE to ODEs along characteristic curves

The previous identity suggests a strategy: instead of trying to solve for $u(x,t)$ everywhere at once, we try to trace curves in the $(x,t)$-plane along which the PDE becomes an ODE. Let's take an example of the quasi-linear PDE

$$
u_t + a(x,t,u)\,u_x = b(x,t,u),
\qquad u(x,0)=\phi(x).
$$

Pick a starting point on the initial line, say $x(t=0)=\xi$ then $u(\xi,0)=\phi(\xi)$. Now, look for a curve $t\mapsto (x(t),t)$ and the value of the solution along it, $t\mapsto u(t):=u(x(t),t)$, such that the PDE holds along that curve. Using the chain rule,

$$
\frac{d}{dt}u(x(t),t)=u_t(x(t),t)+x'(t)\,u_x(x(t),t).
$$

If we choose the curve so that

$$
x'(t)=a\big(x(t),t,u(t)\big),
$$

then substituting into the chain rule and using the PDE gives

$$
\frac{d}{dt}u(t)=b\big(x(t),t,u(t)\big),
\qquad u(0)=\phi(\xi).
$$

So each initial point $\xi$ generates a pair of coupled ODEs,

$$
\begin{cases}
x'(t)=a(x,t,u), & x(0)=\xi,\\
u'(t)=b(x,t,u), & u(0)=\phi(\xi).
\end{cases}
$$

Solving the characteristic ODEs does not immediately give $u$ as an explicit function of $(x,t)$.
Instead it gives a family of curves and values, indexed by the starting point $(\xi)$ on the initial line $(t=0)$. For each fixed $\xi$ we solve the ODEs and obtain

$$
t\mapsto x(t;\xi), \qquad t\mapsto u(t;\xi),
$$

with initial values

$$
x(0;\xi)=\xi, \qquad u(0;\xi)=\phi(\xi).
$$

Thus, for a given $\xi$, the curve $t \longmapsto (x(t;\xi),\,t)$ 
is a characteristic in the $(x,t)$-plane and $u(t;\xi)$ is the value of the solution along that characteristic. To convert this Lagrangian/parametric description into the usual Eulerian form $u=u(x,t)$, we must be able to solve for the label $\xi$ from the relation

$$
x = x(t;\xi).
$$

Equivalently, we need the map

$$
(\xi,t)\ \longmapsto\ (x(t;\xi),\,t)
$$

to be locally invertible, so that $\xi=\xi(x,t)$ and then

$$
u(x,t)=u\bigl(t;\xi(x,t)\bigr).
$$


In practice this means: for a given point $(x,t)$, there should be exactly one label $\xi$ such that $x=x(t;\xi)$. If such a unique $\xi=\xi(x,t)$ exists, we define the solution by

$$
u(x,t)=u\big(t;\xi(x,t)\big).
$$

If instead two different labels $\xi_1\neq \xi_2$ produce the same point $(x,t)$ (i.e. $x(t;\xi_1)=x(t;\xi_2)$), then two characteristic curves ``arrive'' at the same location carrying possibly different values of $u$, so the PDE cannot have a single-valued classical solution there, this is exactly where crossing characteristics signal the breakdown of smooth solutions (and motivates weak solutions/shocks in conservation laws).

### Example: the transport equation (why characteristics feel natural)

To see the method working in the simplest possible setting, consider the constant-coefficient first-order PDE

$$
u_t + c\,u_x = 0,
\qquad u(x,0)=\phi(x),
$$

where $c$ is a constant (a fixed ``speed''). Here $a(x,t,u)=c$ and $b(x,t,u)=0$, so the characteristic ODEs become

$$
x'(t)=c,\quad x(0)=\xi
\qquad\Rightarrow\qquad
x(t;\xi)=\xi+ct,
$$

and

$$
u'(t)=0,\quad u(0)=\phi(\xi)
\qquad\Rightarrow\qquad
u(t;\xi)=\phi(\xi)\ \ \text{(constant along the curve).}
$$

So each initial point $\xi$ launches a straight line in the $(x,t)$-plane, $x=\xi+ct$, and a constant solution value is simply carried along that line. To write the answer as $u(x,t)$, solve $x=\xi+ct$ for the label: $\xi=x-ct$. Substituting gives the explicit solution

$$
u(x,t)=\phi(x-ct).
$$

This formula is worth pausing on: the entire initial profile $\phi$ is translated to the right if $c>0$ (left if $c<0$) at speed $|c|$, without changing shape. In other words, the PDE does not “create” new values, it moves the data along characteristic lines from the initial points, which is exactly the geometric content hidden inside the derivatives $u_t$ and $u_x$.

## Nonlinearity: when characteristics can collide

The transport example worked smoothly because the characteristic speed $c$ was constant, so straight characteristic lines never intersect. Things become more interesting (and more realistic) when the speed depends on the solution itself. Consider

$$
u_t+u\,u_x=0,\qquad u(x,0)=\phi(x),
$$

the inviscid Burgers equation. Along a characteristic curve $t\mapsto (x(t),t),$ we track the solution value on that curve by

$$
U(t):=u(x(t),t).
$$

Which is the same as the case in the previous section, hence comparing gives: $a(x,t,u)=u$ and $b(x,t,u)=0$, so the pair of characteristic ODEs become

$$
\frac{dx}{dt}=u(t),\qquad \frac{du}{dt}=0.
$$

with initial conditions $x(0)=\xi$ and $u(\xi,0)=\phi(\xi)$. Since $u'(t)=0$, the carried value along each characteristic is constant (vary across different initial points $\xi$):

$$
u(t;\xi)=\phi(\xi).
$$

Then the characteristic curve itself has slope determined by that constant value $\xi$:

$$
x'(t)=\phi(\xi)\quad\Rightarrow\quad x(t;\xi)=x(0;\xi)+\phi(\xi)\,t=\xi+\phi(\xi)\,t.
$$

So different parts of the initial profile move at different speeds: the point starting at $\xi$ moves with speed $\phi(\xi)$. If $\phi$ is larger somewhere, that region travels faster and can catch up to slower regions ahead; mathematically this means that the map $\xi\mapsto x(t;\xi)$ can stop being one-to-one. Differentiating,

$$
\frac{\partial x}{\partial \xi}(t;\xi)=1+\phi'(\xi)\,t.
$$

When this derivative becomes zero for some $(t,\xi)$, two nearby labels $\xi$ get mapped to the same $x$, i.e. characteristics collide. After the earliest collision the PDE cannot have a single-valued smooth solution without changing the notion of solution (this is where shocks/weak solutions enter).

### General first-order PDEs and the full characteristic system

The equations we solved so far had the special quasi-linear form $u_t+a(x,t,u)u_x=b(x,t,u)$. A fully general first-order PDE may depend on $u_x$ and $u_t$ in a nonlinear way,

$$
F(x,t,u,u_x,u_t)=0.
$$

In this setting, choosing $x'(t)$ cleverly is not enough, because the PDE does not give $u_t$ as a single linear combination of $u_x$, instead, the geometry of the constraint $F=0$ tells us how $(x,t,u)$ and the derivatives $(u_x,u_t)$ must move together. The method of characteristics introduces additional unknowns to track the derivatives along a curve:

$$
p(t):=u_x(x(t),t),\qquad q(t):=u_t(x(t),t),
$$

so that along the curve we have the constraint

$$
F\big(x(t),t,u(t),p(t),q(t)\big)=0,
$$

in addition to the chain rule identities (showing how the solution varies along the characteristic curve and the speed of that curve)

$$
\frac{d}{dt}u(t)=u_t(x(t),t)+x'(t)\,u_x(x(t),t)=p(t)\,x'(t)+q(t),\qquad \frac{d}{dt}x(t)=x'(t).
$$

The key idea is to choose $x'(t)$ (hence the curve) so that the constraint $F=0$ remains true as we move. This yields a closed system of ODEs for

$$
(x(t),\,t,\,u(t),\,p(t),\,q(t)),
\qquad p(t)=u_x(x(t),t),\ \ q(t)=u_t(x(t),t).
$$

This is the ``full'' characteristic method: we evolve the point $(x,t,u)$ together with its local slopes $(p,q)$ in a way that keeps us on the solution surface $F=0$. Unlike the initial-value construction, we are not parameterizing characteristics by a starting label $\xi$ here, we simply follow a single characteristic trajectory determined by an initial state for $(x,t,u,p,q)$.

## Moving to several space variables (the step needed for Hamilton-Jacobi)

So far we used one space variable $x$ and one time variable $t$. For Hamilton-Jacobi we almost always need many space variables, so let $x\in\mathbb{R}^n$ and write the unknown as $u(x,t)$. Then the first derivatives are

$$
u_t=\frac{\partial u}{\partial t},\qquad \nabla u=\left(\frac{\partial u}{\partial x_1},\dots,\frac{\partial u}{\partial x_n}\right),
$$

and a general first-order PDE becomes

$$
F\big(x,t,u,\nabla u,u_t\big)=0.
$$

Let $s\in\mathbb{R}$ be a parameter (think of it as ``time along the characteristic'') and consider a characteristic curve $s\longmapsto (x(s),t(s)).$ Along a characteristic, $t(s)$ need not be strictly increasing (it could even stay constant), so we introduce this independent parameter $s$ that always parametrizes progress along the curve and lets us write the characteristic ODEs uniformly.

The method of characteristics still says: solve the PDE by following the curves $(x(s),t(s))$  carrying the solution value $u(s)=u(x(s),t(s))$ along them. But now $x(s)$ is a vector curve and the “slope” variable becomes a vector

$$
p(s):=\nabla_x u(x(s),t(s))\in\mathbb{R}^n,\qquad q(s):=u_t(x(s),t(s))\in\mathbb{R}.
$$

The same chain rule idea becomes

$$
\frac{du}{ds}=p\cdot \frac{dx}{ds}+q\,\frac{dt}{ds},
$$

where $\cdot$ is the dot product. Along the curve the PDE becomes the constraint

$$
F\big(x(s),t(s),u(s),p(s),q(s)\big)=0.
$$


Differentiating this constraint with respect to $s$ gives:

$$
0=\frac{d}{ds}F
=F_x\cdot x' + F_t\,t' + F_u\,u' + F_p\cdot p' + F_q\,q'.
$$

Then, a standard choice for the curve direction is

$$
\frac{dx}{ds}= x'=F_p\in\mathbb{R}^n,\qquad \frac{dt}{ds}=t'=F_q\in\mathbb{R},
$$

and the remaining ODEs for $(u,p,q)$ given by

$$
u'=p\cdot x' + q\,t' = p\cdot F_p + q\,F_q.
$$

$$
p'=-F_x - F_u\,p,\qquad q'=-F_t - F_u\,q.
$$

evolve  so that the constraint $F=0$ stays true along the curve. The key point is: even when $x\in\mathbb{R}^n$, a characteristic is still a single parametrized path $s\longmapsto (x(s),t(s))$,
so it's still $1$-dimensional. To actually solve the PDE, we lift this path to the variables

$$
s\longmapsto \big(x(s),t(s),u(s),p(s),q(s)\big),
$$

and evolve them by their respective characteristic ODEs.  The vector $p(s)=\nabla_x u(x(s),t(s))$
records the spatial slopes of $u$ along the path, allowing the PDE condition $F=0$
to determine how $(x,t,u,p,q)$ move together.

### The Hamilton-Jacobi equation as a special first-order PDE

With the multi-variable setup in place, we can now introduce the main object. The Hamilton-Jacobi (HJ) equation is a first-order PDE for a scalar function $S(x,t)$ (often called the action or phase) of the form

$$
S_t + H\!\big(x,\nabla S,t\big)=0,
\qquad x\in\mathbb{R}^n,
$$

together with an initial condition such as

$$
S(x,0)=S_0(x).
$$

Here $H$ is a given function (the Hamiltonian), and the PDE is typically nonlinear because $\nabla S$ appears inside $H$.

### Characteristics of Hamilton-Jacobi

Take the Hamilton-Jacobi PDE

$$
S_t+H(x,\nabla S,t)=0,
\qquad p:=\nabla S,\ q:=S_t,
$$

and rewrite it as the constraint

$$
F(x,t,S,p,q):=q+H(x,p,t)=0.
$$

Now plug this specific $F$ into the symmetric characteristic system

$$
\frac{dx}{ds}=F_p,\quad \frac{dt}{ds}=F_q,\quad \frac{dS}{ds}=p\cdot F_p+qF_q,\quad
\frac{dp}{ds}=-(F_x+pF_S),\quad \frac{dq}{ds}=-(F_t+qF_S).
$$

Because $F=q+H(x,p,t)$ we have

$$
F_p=H_p,\qquad F_q=1,\qquad F_x=H_x,\qquad F_t=H_t,\qquad F_S=0.
$$

So $dt/ds=1$, meaning we can take $s=t$. The characteristic ODEs become the clean system

$$
\frac{dx}{dt}=H_p(x,p,t),\qquad \frac{dp}{dt}=-\,H_x(x,p,t),
$$

which are exactly Hamilton's equations (with $x$ and $p$ evolving along characteristics). The action value $S$ is then obtained by integrating along the same curve:

$$
\frac{dS}{dt}=p\cdot \frac{dx}{dt}+q
= p\cdot H_p(x,p,t)+q.
$$

Finally, the constraint $F=0$ says $q=-H(x,p,t)$, so

$$
\frac{dS}{dt}=p\cdot H_p(x,p,t)-H(x,p,t).
$$

In other words: to solve the HJ PDE, we (i) solve the ODEs for $(x(t),p(t))$, and then (ii) recover $S$ by a single line integral along those characteristic curves. 
To solve the Hamilton-Jacobi initial value problem by characteristics, we launch one characteristic from each initial point $\xi\in\mathbb{R}^n$ on the surface $t=0$, so $x(0)=\xi$. The initial spatial slope of the solution is fixed by the initial data, giving an initial momentum $p(0)=\nabla S_0(\xi)$, and then $(x(t),p(t))$ evolve according to the characteristic ODEs determined by the Hamiltonian $H$. This evolution defines a flow map $\Phi_t:\xi\mapsto x(t;\xi)$. As long as $\Phi_t$ can be inverted (so each point $x$ comes from a unique label $\xi$), we can reconstruct the PDE solution $S(x,t)$ by taking the value carried along the corresponding characteristic. The classical construction fails when the flow map folds i.e. when two different labels reach the same point, so the solution becomes multi-valued and one needs a generalized notion of solution beyond that time.
