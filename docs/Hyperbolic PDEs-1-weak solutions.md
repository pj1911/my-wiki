# Introduction - weak solution to hyperbolic PDEs

This full chapter goes through neural-network (NN) methods for approximating solutions of partial
differential equations (PDEs), with a special focus on hyperbolic PDEs for the current part.

- Both supervised and unsupervised learning approaches are covered, along with
  their main strengths and limitations.

- Key analytical properties of PDE solutions like regularity, stability, and
  conservation laws, are used to motivate NN architecture design. Here, regularity means how smooth the solution is (and whether it has jumps),
  stability means small changes in inputs or initial data should not cause large changes in the output,
  and conservation laws mean certain physical quantities (like mass or energy) should be preserved over time. Therefore, hyperbolic PDEs are
  particularly challenging because solutions often develop discontinuities and other
  irregular features.
  
- Recent hybrid methods that combine classical numerical schemes with NNs are also
  discussed, with the goal of reducing discretization-induced errors. These are errors introduced when a continuous problem (like a PDE)
  is approximated on a finite grid, time step, or set of sample points. These hybrid methods connect PDE theory (e.g., conservation laws and stability) with NN-based solvers and help NNs approximate the exact (continuous) PDE solution more accurately.

## Partial Differential Equations
Partial differential equations (PDEs) describe many physical and engineering systems, from traffic flow to electromagnetism, so solving them efficiently and accurately matters, in practice, many applications need results
that are trustworthy, and the computation fitting within limited time and memory.

A general time-dependent PDE can be written as

$$
\begin{equation}
\left\lbrace
\begin{aligned}
\partial_t u + Lu &= \xi, && (t,x)\in [0,T]\times \mathcal U,\\
u(0,x) &= u_0(x), && x\in \mathcal U,\\
u(t,x) &= g(t), && (t,x)\in [0,T]\times \partial\mathcal U.
\end{aligned}
\right.
\end{equation}
$$

Here, \(t\in[0,T]\) is the time variable, and \(T>0\) is the final time up to which the evolution is
studied. The spatial variable is \(x\in\mathcal U\subset\mathbb R^d\), where \(\mathcal U\) is the
region of interest (an interval in 1D, an area in 2D, or a volume in 3D), and \(d\) is the number of
spatial dimensions. The set \(\partial\mathcal U\) denotes the boundary of \(\mathcal U\) (endpoints
in 1D, a curve in 2D, a surface in 3D). The unknown \(u(t,x)\) is the quantity being solved for, more generally,

$$
u:[0,T]\times\mathcal U \to \mathbb R^n.
$$

The set \([0,T]\times\mathcal U\) is the space--time domain: the \(\times\) means a Cartesian product, i.e.,
all pairs \((t,x)\) with \(t\in[0,T]\) and \(x\in\mathcal U\). Here, \(u\) can be a scalar (\(n=1\), one value at each \((t,x)\)) or a vector-valued field
(\(n>1\), several coupled values at each \((t,x)\)). Furthermore, \(\partial_t u\) denotes the partial derivative of \(u\) with respect to time:
it measures how \(u(t,x)\) changes as \(t\) varies while keeping \(x\) fixed. The operator \(L\) is a spatial differential operator: for each fixed time \(t\), it takes the spatial function
\(u(t,\cdot):\mathcal U\to\mathbb R^n\) as input and returns another function on \(\mathcal U\), built from derivatives with
respect to \(x\) (and possibly \(x\)-dependent coefficients), i.e. \(L[u(t,\cdot)]:\mathcal U\to\mathbb R^n\).
 For example, \(L\) may contain

- Transport/advection terms (first derivatives), such as \(a(x)\cdot\nabla u\), which move the profile of \(u\)
  through space at a velocity field \(a(x)\).
- Diffusion terms (second derivatives), such as \(\nabla\!\cdot\!\big(\kappa(x)\nabla u\big)\) (often written like \(\kappa(x)\Delta u\) in simple cases),
  which spread and smooth \(u\) in space where the coefficient \(\kappa(x)\) controls how strong this smoothing is.
- Reaction/zeroth-order terms (no derivatives), such as \(c(x)u\), which locally scale, damp, or amplify \(u\)
  without moving it in space.

The term \(\xi(t,x)\) is an external input (a source/forcing): it represents effects not captured by transport, diffusion, or other spatial interactions.
At each point \((t,x)\) it directly changes the value of \(u(t,x)\) by adding (a source) or removing (a sink) quantity. For example, if \(u(t,x)\) is temperature, then \(\xi(t,x)\) can represent a heater embedded in the material:
\(\xi(t,x)>0\) where the heater is active (heat is added there) and \(\xi(t,x)<0\) where a cooling device removes heat. In other words, \(\xi\) is a modeled exchange with the environment (energy supplied or extracted). The initial condition \(u(0,x)=u_0(x)\) fixes the state at time \(t=0\) for every \(x\in\mathcal U\).
The boundary condition \(u(t,x)=g(t)\) prescribes values of \(u\) on the boundary \(x\in\partial\mathcal U\)
for all \(t\in[0,T]\).

## Solution of PDEs

A function \(u\) is called a solution of the PDE if it satisfies the PDE together with the initial and
boundary conditions in an appropriate mathematical sense. For smooth solutions, \(u\), the needed derivatives (like \(\partial_t u\) and the spatial derivatives inside \(Lu\))
exist in the usual calculus sense, so the PDE holds at each point \((t,x)\), this is a
classical solution.

### Weak solutions

In hyperbolic PDEs where \(u\) is not smooth (for example, it has jump discontinuities), pointwise derivatives such as
\(\partial_t u\) or \(\nabla u\) may not exist. In that case, the PDE is interpreted in a
weak (integral) sense: Instead of requiring the PDE to hold at every single point, it is required to hold after
being integrated against some smooth test functions. This is justifiable because integration only requires \(u\) to be integrable (not differentiable).
Test functions can be chosen to be nonzero only inside a very small space--time region and zero everywhere else, so the integral checks the PDE balance inside that local neighborhood.
Requiring the identity to hold for all such local tests
forces the PDE to hold throughout the whole domain, while avoiding pointwise derivatives at jumps. More concretely, the weak form does not check \(\partial_t u + Lu = \xi\) pointwise, it checks that

$$
\int_{0}^{T}\!\!\int_{\mathcal U} \big(\partial_t u + Lu - \xi\big)\,\varphi \,dx\,dt = 0
\quad \text{for all smooth } \varphi.
$$

Equivalently, we can define the PDE residual \(r:=\partial_t u+Lu-\xi\). The condition

$$
\int_{0}^{T}\!\!\int_{\mathcal U} r\,\varphi \,dx\,dt = 0 \quad \text{for all smooth } \varphi
$$

means: no matter how a smooth weight \(\varphi\) is chosen, the weighted integral of \(r\) is always
zero. Intuitively, this means \(r\) must be zero when viewed through all test-function integrals (so it cannot be
nonzero on any region without being detected by some choice of \(\varphi\)).
Why this is essentially the same as solving the PDE:

- If \(u\) is smooth, integration by parts turns the identity into the pointwise PDE, so nothing
  is lost.
- If \(u\) is not smooth, \(r\) may not make sense pointwise, but the weighted average
  \(\int r\,\varphi\) still makes sense. Requiring the integral to vanish for every test function is a strong and useful condition:
  if the residual were nonzero on any region of positive size, a test function supported there would
  detect it, so the residual cannot ``hide'' anywhere in the domain.

For example: take \(\varphi\) to be a smooth bump that is \(1\) in a small space--time neighborhood and
\(0\) outside a slightly larger neighborhood. Then the integral above mainly tests whether the PDE
balances inside that neighborhood. Varying the bump over all locations and sizes enforces the balance everywhere, while avoiding
undefined pointwise derivatives at jumps. For hyperbolic PDEs, there is an additional issue:
even the definition of a solution must be handled carefully, since shocks can form and
classical derivatives can fail. This motivates weak formulations and, to select the physically
relevant weak solution, entropy conditions.

### Entropy solutions

Weak solutions are often not unique for hyperbolic conservation laws. A standard example is

$$
\partial_t u + \nabla\!\cdot f(u)=0,
$$

where shocks (jumps) can form even from smooth initial data. The weak form alone may admit multiple
candidates \(u\) that all satisfy the integral identity, so an extra rule is needed to pick the
physically relevant one. For this an entropy solution is introduced, it is a weak solution that, in addition to weak solution property, satisfies a family of integral
inequalities of the form \(\partial_t \eta(u)+\nabla\!\cdot q(u)\le 0\) (in the weak sense) for every
convex entropy \(\eta\) and its associated entropy flux \(q\).

Here, an entropy is a convex scalar function \(\eta:\mathbb R\to\mathbb R\) applied to the
state \(u\), where \(u(t,x)\) is a scalar value. An associated entropy flux
is a function \(q:\mathbb R\to\mathbb R^d\) (in \(d\) space dimensions) chosen to be compatible with
the physical flux \(f\). Compatibility means that, for smooth solutions of the conservation law
\(\partial_t u+\nabla\!\cdot f(u)=0\), the quantity \(\eta(u)\) also satisfies a conservation law
\(\partial_t \eta(u)+\nabla\!\cdot q(u)=0\). To relate \(\eta(u)\) and \(q(u)\), assume the 1D conservation law has a smooth solution \(u(t,x)\):

$$
\partial_t u + \partial_x f(u)=0.
$$

Since \(u\) is smooth, the chain rule gives

$$
\partial_x f\big(u(t,x)\big)=f'\big(u(t,x)\big)\,\partial_x u(t,x).
$$

Equivalently,

$$
\frac{\partial}{\partial x} f\big(u(t,x)\big)
=\frac{df}{du}\Big|_{u=u(t,x)}\,\frac{\partial u}{\partial x}(t,x).
$$

We can write it as, \(\partial_x f(u)=f'(u)\,\partial_x u\), so

$$
\partial_t u + f'(u)\,\partial_x u = 0
\quad\Longrightarrow\quad
\partial_t u = -f'(u)\,\partial_x u.
$$

Let \(\eta\) be a smooth entropy and let \(q\) be an (unknown) flux depending only on \(u\).
By the chain rule,

$$
\partial_t \eta(u)=\eta'(u)\,\partial_t u,
\qquad
\partial_x q(u)=q'(u)\,\partial_x u.
$$

Therefore,

$$
\partial_t \eta(u)+\partial_x q(u)
= \eta'(u)\,\partial_t u + q'(u)\,\partial_x u.
$$

Substitute \(\partial_t u = -f'(u)\,\partial_x u\):

$$
\partial_t \eta(u)+\partial_x q(u)
= \eta'(u)\big(-f'(u)\,\partial_x u\big) + q'(u)\,\partial_x u
= \big(q'(u)-\eta'(u)f'(u)\big)\,\partial_x u.
$$

So, if \(q\) is chosen to satisfy

$$
q'(u)=\eta'(u)\,f'(u),
$$

then \(\partial_t \eta(u)+\partial_x q(u)=0\) for smooth solutions. Note, here \(f'(u)\) and \(q'(u)\)
denote derivatives with respect to \(u\). In the scalar case,
these are ordinary derivatives but in multiple spatial dimensions, \(f'(u)\) and \(q'(u)\) are
\(d\)-dimensional vectors. Equivalently, \(q\) can be recovered (up to an additive constant) by

$$
q(u)=\int^{u}\eta'(s)\,f'(s)\,ds.
$$

This derivation shows how to pair an entropy \(\eta\) with the correct entropy flux \(q\): for
smooth solutions, the original conservation law automatically implies a conservation law for
\(\eta(u)\)

$$
\partial_t \eta(u) + \nabla\!\cdot q(u)=0.
$$

This is the starting point for the entropy inequality used when shocks appear,
where the same pair \((\eta,q)\) is kept but the equality is relaxed to select the physically
relevant weak solution. Therefore, across shocks, the weak form alone can accept multiple solutions. The entropy condition
adds an extra rule: pick the solution where entropy does not increase across the shock.

#### Traffic intuition (LWR model)

In \(\partial_t\rho+\partial_x f(\rho)=0\), shocks represent compression (cars brake and bunch
up). Compression can happen abruptly, so a low\(\to\)high density jump can be an admissible shock.
The opposite transition, high\(\to\)low density, is decompression: gaps open up gradually as
information travels through the flow. A sudden high\(\to\)low jump (a rarefaction shock) would
require an instantaneous response, so it is ruled out by the entropy condition. Instead, the model
produces a rarefaction fan, a smooth spreading wave.

**What the entropy inequality adds (and how it matches the intuition).**

The intuition above says: a compressive jump is allowed, but a decompressive jump is not. The
entropy inequality is the mathematical way to enforce that idea. It introduces a scalar
``disorder'' measure \(\eta(\rho)\) (any convex function) and requires that this quantity cannot be
created inside the road, except for what enters through the boundaries. Pick any convex entropy function \(\eta(\rho)\) and its associated entropy flux \(q(\rho)\). An entropy
solution satisfies

$$
\partial_t \eta(\rho) + \partial_x q(\rho)\le 0
\quad\text{(in the weak sense).}
$$

This inequality rules out rarefaction shocks because those jumps would act like an internal
source of entropy; compressive shocks are allowed because they act like dissipation. The inequality is easiest to read on a road segment \([a,b]\). Define the entropy content on \([a,b]\) by

$$
E(t):=\int_a^b \eta(\rho(t,x))\,dx.
$$

Differentiate under the integral sign:

$$
\frac{d}{dt}E(t)=\int_a^b \partial_t \eta(\rho(t,x))\,dx.
$$

Now using the entropy inequality

$$
\partial_t \eta(\rho) + \partial_x q(\rho)\le 0
\quad\Longrightarrow\quad
\partial_t \eta(\rho)\le -\partial_x q(\rho),
$$

we get

$$
\frac{d}{dt}E(t)\le \int_a^b \big(-\partial_x q(\rho(t,x))\big)\,dx.
$$

Finally, apply the fundamental theorem of calculus in \(x\):

$$
\int_a^b -\partial_x q(\rho(t,x))\,dx
= -\Big[q(\rho(t,x))\Big]_{x=a}^{x=b}
= q(\rho(t,a)) - q(\rho(t,b)).
$$

So,

$$
\frac{d}{dt}E(t)\;\le\; q(\rho(t,a)) - q(\rho(t,b)).
$$

This makes the link to the intuition concrete: the only way entropy inside \([a,b]\) can increase is
if more entropy enters at \(x=a\) than leaves at \(x=b\). If the boundaries are not
injecting entropy, then \(E(t)\) cannot increase, meaning any shock inside the segment must be
entropy-dissipating (compressive) rather than entropy-creating (a rarefaction shock).

#### Understanding the inequality

The inequality \(\partial_t\eta(u)+\nabla\!\cdot q(u)\le 0\) is understood in the same weak (integral)
way as before: derivatives are moved onto a smooth test function. Concretely, for every smooth
nonnegative test function \(\varphi(t,x)\ge 0\) (often taken to vanish on the boundary and at
\(t=T\)), start from the formal step (valid for smooth \(u\))

$$
\int_{0}^{T}\!\!\int_{\mathcal U}\big(\partial_t\eta(u)+\nabla\!\cdot q(u)\big)\,\varphi \,dx\,dt \le 0.
$$

A practical way to derive the weak (integral) identities is to start from a pointwise product rule,
integrate it, and then rearrange terms.

#### Integration by parts in time

**Step 1 (pointwise identity).**
For smooth \(a(t,x)\) and \(\varphi(t,x)\),

$$
\partial_t(a\varphi)= (\partial_t a)\,\varphi + a\,\partial_t\varphi.
$$

**Step 2 (integrate the identity).**
Since the equality holds at every point \((t,x)\), integrate both sides over the full
space--time domain \([0,T]\times\mathcal U\):

$$
\int_{0}^{T}\int_{\mathcal U}\partial_t\big(a\varphi\big)\,dx\,dt = \int_{0}^{T}\int_{\mathcal U}(\partial_t a)\,\varphi\,dx\,dt
+
\int_{0}^{T}\int_{\mathcal U}a\,\partial_t\varphi\,dx\,dt.
$$

This turns a pointwise statement into a statement about averaged (integrated) quantities, which is
the form needed later when derivatives of \(u\) may not exist pointwise.


**Step 3 (evaluate the left-hand side).**
Focus on

$$
\int_{0}^{T}\!\!\int_{\mathcal U}\partial_t(a\varphi)\,dx\,dt.
$$

First, fix \(x\) and integrate in time:

$$
\int_{0}^{T}\partial_t(a\varphi)(t,x)\,dt = (a\varphi)(T,x)-(a\varphi)(0,x),
$$

by the fundamental theorem of calculus. Now integrate this result over \(x\in\mathcal U\):

$$
\int_{0}^{T}\!\!\int_{\mathcal U}\partial_t(a\varphi)\,dx\,dt = \int_{\mathcal U}(a\varphi)(T,x)\,dx-\int_{\mathcal U}(a\varphi)(0,x)\,dx =
\Big[\int_{\mathcal U}a\,\varphi\,dx\Big]_{t=0}^{t=T}.
$$

**Step 4 (rearrange to isolate the term with \(\partial_t a\)).**
Combine Step 2 and Step 3:

$$
\Big[\int_{\mathcal U}a\,\varphi\,dx\Big]_{t=0}^{t=T}
=\int_{0}^{T}\!\!\int_{\mathcal U}(\partial_t a)\,\varphi\,dx\,dt
+\int_{0}^{T}\!\!\int_{\mathcal U}a\,\partial_t\varphi\,dx\,dt.
$$

Now subtract the last integral from both sides:

$$
\int_{0}^{T}\!\!\int_{\mathcal U}(\partial_t a)\,\varphi\,dx\,dt
=\Big[\int_{\mathcal U}a\,\varphi\,dx\Big]_{t=0}^{t=T}
-\int_{0}^{T}\!\!\int_{\mathcal U}a\,\partial_t\varphi\,dx\,dt.
$$

This is the time integration-by-parts formula used in the weak form.

#### Integration by parts in space

**Step 1 (pointwise identity).**
For a smooth vector field \(F(x)\) and a smooth scalar test function \(\varphi(x)\),

$$
\nabla\!\cdot(\varphi F)= (\nabla\varphi)\cdot F + \varphi\,(\nabla\!\cdot F).
$$

**Step 2 (integrate the identity over space).**
Integrate both sides over \(x\in\mathcal U\):

$$
\int_{\mathcal U}\nabla\!\cdot(\varphi F)\,dx =
\int_{\mathcal U}(\nabla\varphi)\cdot F\,dx
+
\int_{\mathcal U}\varphi\,(\nabla\!\cdot F)\,dx.
$$

**Step 3 (apply the divergence theorem to the left-hand side).**
The divergence theorem says

$$
\int_{\mathcal U}\nabla\!\cdot(\varphi F)\,dx =
\int_{\partial\mathcal U}\varphi\,F\cdot n\,dS,
$$

where \(n(x)\) is the outward unit normal vector on the boundary \(\partial\mathcal U\), and \(dS\) is
the surface measure on \(\partial\mathcal U\) (a length element in 2D, a surface-area element in 3D).

**Step 4 (rearrange to isolate the term with \(\nabla\!\cdot F\)).**
Combine Step 2 and Step 3:

$$
\int_{\partial\mathcal U}\varphi\,F\cdot n\,dS =
\int_{\mathcal U}(\nabla\varphi)\cdot F\,dx
+
\int_{\mathcal U}\varphi\,(\nabla\!\cdot F)\,dx.
$$

Now move the \((\nabla\varphi)\cdot F\) term to the other side:

$$
\int_{\mathcal U}\varphi\,(\nabla\!\cdot F)\,dx = \int_{\partial\mathcal U}\varphi\,F\cdot n\,dS -
\int_{\mathcal U}F\cdot\nabla\varphi\,dx.
$$

**Step 5 (extend to space-time).**
Apply the spatial identity at each fixed time \(t\), then integrate the whole equation over \(t\in[0,T]\):

$$
\int_{0}^{T}\int_{\mathcal U}\varphi\,(\nabla\!\cdot F)\,dx\,dt =
\int_{0}^{T}\int_{\partial\mathcal U}\varphi\,F\cdot n\,dS\,dt -
\int_{0}^{T}\int_{\mathcal U}F\cdot\nabla\varphi\,dx\,dt.
$$

In the entropy setting, the choice is \(F=q(u)\), so

$$
\int_{0}^{T}\int_{\mathcal U}(\nabla\!\cdot q(u))\,\varphi\,dx\,dt =
\int_{0}^{T}\int_{\partial\mathcal U}\varphi\,q(u)\cdot n\,dS\,dt -
\int_{0}^{T}\int_{\mathcal U}q(u)\cdot\nabla\varphi\,dx\,dt.
$$

**Step 6 (derive the weak entropy inequality).**

Start from the pointwise entropy inequality (formal, for smooth \(u\)):

$$
\partial_t\eta(u)+\nabla\!\cdot q(u)\le 0.
$$

Multiply by a smooth nonnegative test function \(\varphi\ge 0\) and integrate over
\([0,T]\times\mathcal U\):

$$
\int_{0}^{T}\int_{\mathcal U}\big(\partial_t\eta(u)+\nabla\!\cdot q(u)\big)\,\varphi\,dx\,dt \le 0.
$$

Now apply the two integration-by-parts results:

- For the time term \(\int\!\!\int \partial_t\eta(u)\,\varphi\), use the time formula with
  \(a=\eta(u)\).
- For the space term \(\int\!\!\int (\nabla\!\cdot q(u))\,\varphi\), use the space formula with
  \(F=q(u)\).

**Step 7 (Substituting the values)**

Start from

$$
\int_{0}^{T}\int_{\mathcal U}\big(\partial_t\eta(u)+\nabla\!\cdot q(u)\big)\,\varphi\,dx\,dt \le 0,
\qquad \varphi\ge 0.
$$

Split the two terms:

$$
\int_{0}^{T}\int_{\mathcal U}\partial_t\eta(u)\,\varphi\,dx\,dt
\;+\;
\int_{0}^{T}\int_{\mathcal U}(\nabla\!\cdot q(u))\,\varphi\,dx\,dt
\le 0.
$$

**Time term (use time integration by parts with \(a=\eta(u)\)).**

$$
\int_{0}^{T}\int_{\mathcal U}\partial_t\eta(u)\,\varphi\,dx\,dt =
\Big[\int_{\mathcal U}\eta(u)\,\varphi\,dx\Big]_{t=0}^{t=T}
-\int_{0}^{T}\int_{\mathcal U}\eta(u)\,\partial_t\varphi\,dx\,dt.
$$

If \(\varphi(T,\cdot)=0\), then \(\int_{\mathcal U}\eta(u(T,x))\,\varphi(T,x)\,dx=0\), hence

$$
\Big[\int_{\mathcal U}\eta(u)\,\varphi\,dx\Big]_{t=0}^{t=T} =
0-\int_{\mathcal U}\eta(u(0,x))\,\varphi(0,x)\,dx =
-\int_{\mathcal U}\eta(u(0,x))\,\varphi(0,x)\,dx.
$$

Now use the initial condition \(u(0,x)=u_0(x)\), which implies \(\eta(u(0,x))=\eta(u_0(x))\), giving

$$
\Big[\int_{\mathcal U}\eta(u)\,\varphi\,dx\Big]_{t=0}^{t=T} =
-\int_{\mathcal U}\eta(u_0(x))\,\varphi(0,x)\,dx.
$$

Therefore,

$$
\int_{0}^{T}\int_{\mathcal U}\partial_t\eta(u)\,\varphi\,dx\,dt =
-\int_{\mathcal U}\eta(u_0(x))\,\varphi(0,x)\,dx
-\int_{0}^{T}\int_{\mathcal U}\eta(u)\,\partial_t\varphi\,dx\,dt.
$$

**Space term (use spatial integration by parts with \(F=q(u)\)).**

$$
\int_{0}^{T}\int_{\mathcal U}(\nabla\!\cdot q(u))\,\varphi\,dx\,dt =
\int_{0}^{T}\int_{\partial\mathcal U}\varphi\,q(u)\cdot n\,dS\,dt
-\int_{0}^{T}\int_{\mathcal U}q(u)\cdot\nabla\varphi\,dx\,dt.
$$

If \(\varphi=0\) on \(\partial\mathcal U\), then the boundary integral is \(0\), so

$$
\int_{0}^{T}\int_{\mathcal U}(\nabla\!\cdot q(u))\,\varphi\,dx\,dt =
-\int_{0}^{T}\int_{\mathcal U}q(u)\cdot\nabla\varphi\,dx\,dt.
$$

**Put both pieces back into the inequality.**

$$
-\int_{\mathcal U}\eta(u_0(x))\,\varphi(0,x)\,dx
-\int_{0}^{T}\int_{\mathcal U}\eta(u)\,\partial_t\varphi\,dx\,dt
-\int_{0}^{T}\int_{\mathcal U}q(u)\cdot\nabla\varphi\,dx\,dt
\le 0.
$$

Rearranging,

$$
\int_{\mathcal U}\eta(u_0(x))\,\varphi(0,x)\,dx
+\int_{0}^{T}\int_{\mathcal U}\eta(u)\,\partial_t\varphi\,dx\,dt
+\int_{0}^{T}\int_{\mathcal U}q(u)\cdot\nabla\varphi\,dx\,dt
\ge 0,
$$

i.e.

$$
\int_{0}^{T}\int_{\mathcal U}
\Big(\eta(u)\,\partial_t \varphi + q(u)\cdot \nabla \varphi \Big)\,dx\,dt
+ \int_{\mathcal U}\eta\big(u_0(x)\big)\,\varphi(0,x)\,dx \ge 0.
$$

The goal of these steps is to rewrite the entropy inequality in a form that
avoids pointwise derivatives of \(u\). The final statement is an integral condition involving
only \(\eta(u)\) and \(q(u)\) tested against smooth \(\varphi\), so it still makes sense when \(u\) has
jumps. This is the standard definition of an entropy solution: a weak solution that also satisfies
this inequality for all convex entropies (and all nonnegative tests). It is also the form used in
analysis (for uniqueness/stability) and in practice (to design entropy-stable numerical schemes and
learning losses that enforce the correct solution selection).

At this point, two natural technical questions come up about the choice of test function \(\varphi\): 

**1) Why require \(\varphi\ge 0\)?**
Because the starting statement is an inequality (not an equality). Testing it against all
nonnegative \(\varphi\) is the standard weak way to encode '\(\le 0\)' without needing pointwise
derivatives.

**2) What if \(\varphi\) does not vanish on \(\partial\mathcal U\) or at \(t=T\)?**
Then the boundary terms kept earlier stay in the formula:

$$
\Big[\int_{\mathcal U}\eta(u)\,\varphi\,dx\Big]_{t=0}^{t=T}
+
\int_{0}^{T}\int_{\partial\mathcal U}\varphi\,q(u)\cdot n\,dS\,dt
$$

and the precise entropy condition depends on how boundary data are imposed. Setting
\(\varphi(T,\cdot)=0\) and \(\varphi|_{\partial\mathcal U}=0\) is mainly a clean choice to present the
core idea without extra terms.

## Wrapping up
Connection to the weak formulation:

- The weak form encodes the PDE by integrated balances, so it remains meaningful even
  when \(u\) has jumps.
- For hyperbolic problems, the weak form can admit multiple solutions. The
  entropy condition adds an extra inequality that selects the physically relevant one
  (and typically gives uniqueness within the right class).
- Many numerical methods are built to be entropy stable, so their discrete solutions
  respect the same selection principle and avoid converging to a wrong weak solution.

Intuition: a shock can satisfy the conservation law in an averaged sense, yet still be assembled in
more than one mathematically valid way. Entropy rules out the nonphysical shock patterns by
enforcing the correct direction of information flow and the expected dissipation across the jump.
