## Physical setting and notation

We consider traffic on a single lane, represented as a one-dimensional continuum along a road coordinate \(x\in\mathbb{R}\) and time \(t\ge 0\).

**Macroscopic state variables.**

- \(\rho(x,t)\ge 0\): vehicle density (vehicles per unit length).
- \(v(x,t)\ge 0\): mean vehicle speed (length per unit time).
- \(q(x,t) := \rho(x,t)v(x,t)\): traffic flow (vehicles per unit time).

**Constitutive functions.**

- \(p(\rho)\): an increasing function of density (often called traffic pressure or hesitation/anticipation function). A standard assumption is \(p(0)=0\) and \(p'(\rho)>0\) for \(\rho>0\) [[4](#ref-4), [3](#ref-3)].
- \(V(\rho)\): an equilibrium speed curve (decreasing with density in typical calibrations) [[3](#ref-3)].
- \(\tau>0\): a relaxation time scale governing adaptation toward equilibrium [[3](#ref-3)].

## Conservation of vehicles

Conservation of the number of cars in any road segment \([a,b]\) implies

$$
\frac{\mathrm{d}}{\mathrm{d} t}\int_a^b \rho(x,t)\,\mathrm{d} x
\;=\;
q(a,t)-q(b,t).
$$

Assuming sufficient regularity and using the divergence theorem in 1D yields the continuity equation

<a id="eq-continuity"></a>

$$
\frac{\partial \rho}{\partial t}
\;+\;
\frac{\partial}{\partial x}\big(\rho v\big)
\;=\;
0.
$$

### Beyond a single conservation law

Equation [\((2)\)](#eq-continuity) is a balance law for the density \(\rho\), but it is not closed.
By closed we mean: the PDE system contains enough relations to determine all unknown fields from initial/boundary data.
Here we have two unknowns \((\rho,v)\), while [\((2)\)](#eq-continuity) provides only one equation. Equivalently, the flux

$$
q(\rho,v):=\rho\,v
$$

cannot be computed from \(\rho\) alone unless we specify an additional relation for \(v\).

**LWR closure.** The Lighthill--Whitham--Richards (LWR) model closes [\((2)\)](#eq-continuity) by postulating an equilibrium
velocity--density relation

$$
v = V(\rho),
$$

where \(V(\rho)\) is a prescribed decreasing function (the fundamental diagram). Then the flux becomes a function
of \(\rho\) only,

$$
q(\rho)=\rho\,V(\rho),
$$

and [\((2)\)](#eq-continuity) reduces to the scalar conservation law

$$
\rho_t + \big(\rho V(\rho)\big)_x = 0.
$$

**Why go beyond LWR?** LWR enforces \(v=V(\rho)\) pointwise (instantaneous equilibrium). To model non-equilibrium effects (e.g.\ delayed
acceleration, anticipation, heterogeneous driving), we keep \(v\) (or a related driving state) as an additional
unknown and supply a second evolution equation.

This leads to second--order traffic models: systems of PDEs that keep the conservation law for \(\rho\) but add a
second equation encoding how the velocity (or a modified velocity) is transported/updated. Our next step will be to
introduce such a model called the ARZ model and then justify its form.

## ARZ model: closing the system with directional interaction

Second-order macroscopic traffic models close the conservation law [\((2)\)](#eq-continuity) by introducing an additional
state variable that captures non--equilibrium driving effects. The motivation is twofold:

- Non-equilibrium speeds: the velocity \(v\) is not assumed to satisfy an instantaneous relation
  \(v=V(\rho)\) as in LWR, instead, drivers adjust their speed over a finite time scale (acceleration/braking limits and reaction time), so \(v\) has its own dynamics.
- Heterogeneous drivers: even at the same density \(\rho\), different drivers may have different
  preferred behavior. This requires an additional field beyond \(\rho\).

### Directional (anisotropic) interaction as a closure principle.

A key empirical feature of traffic is that drivers primarily react to conditions ahead.
At the PDE level, look ahead only means a one--sided domain of dependence: for any given time \(t\), the state at \((t,x)\) should be
determined only by data (vehicles) located at positions \(y\ge x\) (ahead on road), and not by data (vehicles) from \(y<x\) (behind on road).

#### One-sided domain of dependence (proof)

To prove the above statement we consider the linear transport equation

<a id="eq-linear-transport"></a>

$$
q_t + a(t,x)\,q_x = 0 \qquad \text{on } (0,T)\times\mathbb{R}.
$$

with initial data \(q(0,x)=q_0(x)\). Assume \(a\in C^1\) and

$$
a(t,x)\le 0 \quad \text{for all } (t,x)\in(0,T)\times\mathbb{R}.
$$

Here \(q(t,x)\) denotes the transported scalar quantity (the unknown), and \(a(t,x)\) is a given transport
velocity field prescribing the local propagation speed in the \(x\)-direction. \(a(t,x)\le 0\) encodes a left-moving transport, so information propagates from \(y\ge x\) (ahead) toward \(x\) (backward), which is exactly the look ahead influence we want to model.

**Characteristic ODE.** We define the characteristic ODE associated with [\((7)\)](#eq-linear-transport) as follows.
A characteristic is a curve \(s\mapsto (s,X(s))\) in the \((t,x)\) plane along which we track \(q\).
Setting \(\psi(s):=q(s,X(s))\), the chain rule gives

$$
\psi'(s)=q_t(s,X(s)) + X'(s)\,q_x(s,X(s)).
$$

To make this derivative match the PDE expression \(q_t+a\,q_x\), we choose the curve so that

$$
X'(s)=a\big(s,X(s)\big),
$$

which is the ODE we refer to as the characteristic ODE. Here \(s\) is the curve parameter (time) running in \([0,t]\).

**Claim.** Assume \(q\in C^1\) solves \(q_t+a(t,x)q_x=0\) on \((0,T)\times\mathbb{R}\) with initial data \(q(0,x)=q_0(x)\), and assume
\(a\in C^1\) with \(a(t,x)\le 0\).

Fix \((t,x)\in(0,T)\times\mathbb{R}\) and let \(X(\cdot)\) be the (unique) \(C^1\) solution of

$$
\dot X(s)=a\big(s,X(s)\big),\qquad X(t)=x,\qquad s\in[0,t].
$$

Then we prove:

- (i) value is transported from the initial line: \(q(t,x)=q_0\big(X(0)\big)\);
- (ii) footpoint lies ahead: \(X(0)\ge x\), and therefore \(q(t,x)\) can only depend on initial data
  from positions \(y\ge x\).

**Proof.** Fix \((t,x)\) and let \(X(\cdot)\) solve \(\dot X(s)=a(s,X(s))\) with \(X(t)=x\). Define the restriction of \(q\) to this curve by

$$
\psi(s):=q\big(s,X(s)\big),\qquad s\in[0,t].
$$

By the chain rule,

$$
\psi'(s)=q_t\big(s,X(s)\big)+\dot X(s)\,q_x\big(s,X(s)\big).
$$

Substituting \(\dot X(s)=a\big(s,X(s)\big)\) gives

$$
\psi'(s)=q_t\big(s,X(s)\big)+a\big(s,X(s)\big)\,q_x\big(s,X(s)\big).
$$

Since \(q\) solves \(q_t+a q_x=0\), evaluating the PDE at the point \((s,X(s))\) yields \(\psi'(s)=0\) for all \(s\in[0,t]\).
Therefore \(\psi\) is constant, so \(\psi(t)=\psi(0)\), i.e.

$$
q\big(t,X(t)\big)=q\big(0,X(0)\big).
$$

Using \(X(t)=x\) and the initial condition \(q(0,\cdot)=q_0(\cdot)\), we obtain

$$
q(t,x)=q_0\big(X(0)\big).
$$

Finally, if \(a\le 0\) then \(\dot X(s)=a(s,X(s))\le 0\) for all \(s\), so \(X\) is nonincreasing on \([0,t]\). Hence

$$
X(0)\ge X(t)=x,
$$

which shows the footpoint \(X(0)\) lies ahead of \(x\). Because the influencing point \(X(0)\) lies ahead of \(x\) (\(y\ge x\)), data from behind (\(y<x\)) cannot affect \(q(t,x)\).

### The Aw--Rascle--Zhang (ARZ) anisotropic closure.

In the previous section, we wanted to introduce an additional state variable to capture non-equilibrium effects. At the PDE level, introducing a new state variable forces a new evolution law: to have a closed system,
we must specify how that variable changes in time and space, i.e.\ we must add a second equation. The one-sided domain-of-dependence proof gives a concrete PDE test for look-ahead only. We can now use this as a closure principle: we will choose the second equation so that the
additional driver-dependent information is advected in the correct (anisotropic) direction.

Before formulating an equation to close the system, we introduce an additional field \(w\) often called the Lagrangian marker, representing a driver-dependent
preferred speed level not fixed by \(\rho\) alone, by defining

<a id="eq-marker-def"></a>

$$
w := v + P(\rho).
$$

where \(P(\rho)\) is a monotone traffic pressure with \(P'(\rho)>0\), \(\rho(t,x)\) is the vehicle density and \(v(t,x)\) is the actual macroscopic speed. So if a given vehicle carries a fixed \(w\) value, then when it enters a region of higher density (larger \(P(\rho)\)) its speed decreases accordingly. Conversely, in lighter traffic it can travel faster, up to the level permitted by its marker. We chose this \(w\) to represent a driver/vehicle attribute (preferred speed level or aggressiveness) that
does not instantly equilibrate with density: it is tied to the vehicles themselves rather than created or destroyed by the flow.
Hence, if we follow a specific vehicle (a trajectory \(x=X(t)\) with \(\dot X(t)=v(t,X(t))\)), we postulate that this attribute is preserved:

$$
\frac{d}{dt}w\big(t,X(t)\big)=0.
$$

Applying the chain rule gives

$$
\begin{aligned}
0 &= \frac{d}{dt}w\big(t,X(t)\big) \\
0 &= w_t\big(t,X(t)\big)+\dot X(t)\,w_x\big(t,X(t)\big) \\
0 &= w_t\big(t,X(t)\big)+v\big(t,X(t)\big)\,w_x\big(t,X(t)\big),
\end{aligned}
$$

which yields the PDE

<a id="eq-w-transport"></a>

$$
w_t+v\,w_x=0.
$$

This means \(w\) moves with vehicles, so information in \(w\)
propagates along vehicle paths. The point of imposing

$$
\begin{aligned}
\frac{d}{dt}w\big(t,X(t)\big) &= 0 \\
\qquad\text{along } \dot X(t) &= v(t,X(t))
\end{aligned}
$$

is that \(w\) is meant to encode a driver-specific preference that should not be altered by surrounding traffic. If \(w\) is carried by vehicles, then \(w\) is constant along each vehicle path \(X(\cdot)\). So the value \(w(t,x)\) is inherited from the position \(X(0)\) of the same vehicle at the initial time.
If the characteristic/vehicle paths satisfy \(X(0)\ge x\) (the one-sided dependence property), then the only possible
source point \(X(0)\) lies at positions \(y\ge x\) (ahead), and data from \(y<x\) (behind) cannot influence \(w(t,x)\).

Once we decide (i) what the extra state variable is (\(w:=v+P(\rho)\)) and (ii) how it evolves (it is advected with speed \(v\)),
there is no further freedom: the closure law the becomes

$$
\begin{aligned}
w_t+v w_x &= 0 \\
(\,v+P(\rho)\,)_t+v(\,v+P(\rho)\,)_x &= 0.
\end{aligned}
$$

This gives us the homogeneous ARZ system in non-conservative form:

<a id="eq-arz-nonconservative"></a>

$$
\begin{aligned}
\rho_t+(\rho v)_x &= 0, \\
(\,v+P(\rho)\,)_t+v(\,v+P(\rho)\,)_x &= 0,
\end{aligned}
\qquad P\in C^1.
$$

We now have two equations: (i) mass conservation for \(\rho\), and (ii) the closure law stating that \(w\) is transported with the vehicle speed \(v\). We can now rewrite the system in conservative form:

$$
\begin{aligned}
\rho_t+(\rho v)_x &= 0, \\
w_t+v w_x &= 0.
\end{aligned}
$$

Multiply the first by \(w\) and the second by \(\rho\), then add:

$$
w\big(\rho_t+(\rho v)_x\big)+\rho\big(w_t+v w_x\big)=0.
$$

Expand the left-hand side:

$$
w\rho_t+w(\rho v)_x+\rho w_t+\rho v w_x=0.
$$

Now recognize the two grouped terms as product derivatives:

$$
\begin{aligned}
w\rho_t+\rho w_t &= (\rho w)_t, \\
w(\rho v)_x+\rho v w_x &= (\rho v w)_x.
\end{aligned}
$$

Substitute these identities to get

$$
(\rho w)_t+(\rho v w)_x=0.
$$

We see that combining the two relations allows us to rewrite the closure law in conservative form: rather than evolving \(w\) directly, we obtain a conservation law for the quantity \(\rho w\) with flux \(\rho v w\). Together with mass conservation \(\rho_t+(\rho v)_x=0\), this gives the homogeneous ARZ system.

<a id="eq-arz-conservative"></a>

$$
\begin{aligned}
\rho_t + (\rho v)_x &= 0, \\
(\rho w)_t + (\rho v w)_x &= 0,
\end{aligned}
\qquad \text{with } v = w - P(\rho).
$$

### Why this is the correct look-ahead closure (characteristic-speed derivation).

Consider the homogeneous non-conservative ARZ system given by [\((24)\)](#eq-arz-nonconservative):

$$
\begin{aligned}
\rho_t+(\rho v)_x&=0, \\
(\,v+P(\rho)\,)_t+v(\,v+P(\rho)\,)_x&=0,
\end{aligned}
$$

with \(P\in C^1\).

**Step 1: Quasilinear form in \((\rho,v)\).** Expand the continuity equation:

$$
\rho_t+v\rho_x+\rho v_x=0.
$$

Expand the Lagrangian marker equation using the chain rule:

$$
v_t+P'(\rho)\rho_t+v v_x+v P'(\rho)\rho_x=0.
$$

Substitute \(\rho_t=-v\rho_x-\rho v_x\) from continuity:

$$
0
= v_t+P'(\rho)\big(-v\rho_x-\rho v_x\big)+v v_x+v P'(\rho)\rho_x ,
$$

<a id="eq-v-quasilinear"></a>

$$
0=v_t+\big(v-\rho P'(\rho)\big)v_x .
$$

since the \(\rho_x\) terms cancel. Hence the system can be written as the quasilinear system

$$
\begin{pmatrix}\rho\\ v\end{pmatrix}_t
+
\underbrace{\begin{pmatrix}
v & \rho\\
0 & v-\rho P'(\rho)
\end{pmatrix}}_{A(\rho,v)}
\begin{pmatrix}\rho\\ v\end{pmatrix}_x
=0.
$$

**Step 2: Eigenvalues = characteristic speeds.** For \(U_t+A(U)U_x=0\), characteristic speeds are the eigenvalues of \(A(U)\).
Here \(A(\rho,v)\) is upper triangular, so its eigenvalues are its diagonal entries:

$$
\lambda_2=v,\qquad \lambda_1=v-\rho P'(\rho).
$$

**Step 3: Consequence (anisotropy / look-ahead).** Assuming \(P'(\rho)\ge 0\) and using the road orientation adopted previously (vehicles move in the negative \(x\)-direction,
so \(v\le 0\)), then

$$
\lambda_1\le \lambda_2\le 0.
$$

Thus both characteristic families propagate from larger positions to smaller positions, so tracing characteristics
back to the initial line gives footpoints \(X(0)\ge x\). Therefore the state at \((t,x)\) can only depend on initial data
from positions \(y\ge x\) (ahead), never from \(y<x\) (behind): this is the look-ahead only property.

## The ARZ model with relaxation toward equilibrium speed

Macroscopic traffic models often distinguish between:
(i) an equilibrium or fundamental relation between speed and density, denoted by \(v = V(\rho)\).
(ii) a macroscopic velocity field \(v(x,t)\), interpreted as the local (coarse-grained) mean vehicle speed near position \(x\) at time \(t\). Empirically, under approximately steady conditions this mean speed is often well described by a decreasing equilibrium function \(v=V(\rho)\). The first-order LWR model enforces this equilibrium relation as a constraint at all times, \(v(x,t)=V(\rho(x,t))\). In contrast, second-order models such as ARZ treat \(v\) as an independent state variable, so \((\rho,v)\) need not lie on the manifold \(v=V(\rho)\) pointwise. Instead, \(V(\rho)\) serves as a reference equilibrium that the dynamics may approach (typically via relaxation) rather than a relation that is imposed identically.

### Relaxation time \((\tau)\).

Define a parameter \(\tau>0\) as relaxation time: it sets how quickly drivers adjust their actual speed toward the equilibrium speed.
A standard modeling choice is a linear relaxation law,

$$
\text{(rate of change of speed)} \;\propto\; V(\rho)-v,
$$

so that if \(v\) is below \(V(\rho)\) drivers accelerate, and if \(v\) is above \(V(\rho)\) drivers decelerate.
Introducing the proportionality constant \(1/\tau\) yields the source term

$$
\frac{V(\rho)-v}{\tau}.
$$

This has units of acceleration (speed divided by time). Moreover, if density is held fixed and other transport effects are ignored,
the local speed relaxes exponentially to equilibrium:

$$
\begin{aligned}
\frac{\partial v}{\partial t}&=\frac{V(\rho)-v}{\tau} \\
v(t)-V(\rho)&=\big(v(0)-V(\rho)\big)e^{-t/\tau}.
\end{aligned}
$$

Thus \(\tau\) is the characteristic time after which the deviation from equilibrium has decreased by a factor of \(e\). To model relaxation toward an equilibrium speed \(V(\rho)\), a common (and widely used) conservative and inhomogeneous ARZ form in variables \((\rho,v)\) is [[3](#ref-3)] (shown in [\((30)\)](#eq-arz-conservative) without the forcing term, substituting \( w\rho = y\)):

<a id="eq-arz-conservative2"></a>

$$
\begin{aligned}
\rho_t + (\rho v)_x &= 0,\\
(y)_t + (y v )_x &= \rho\,\dfrac{V(\rho)-v}{\tau},
\end{aligned}
\qquad \text{with } v = w - P(\rho).
$$

### Justification by the local relaxation principle.

If we ignore transport effects and hold density fixed locally (formally: drop all \(x\)-derivatives),
then [\((42)\)](#eq-arz-conservative2) reduces to

$$
\frac{\partial \rho}{\partial t}=0
\quad\Rightarrow\quad
\rho(t)\equiv \rho_0,
\qquad
\frac{\partial y}{\partial t}=\rho_0\,\frac{V(\rho_0)-v}{\tau}.
$$

But \(y=\rho w = \rho(v+p(\rho))\), and with \(\rho\) constant in time, \(p(\rho)\) is also constant, so

$$
\frac{\partial y}{\partial t}=\rho_0\,\frac{\partial v}{\partial t}.
$$

Hence the second equation gives exactly the desired ODE

$$
\frac{\partial v}{\partial t}=\frac{V(\rho_0)-v}{\tau},
$$

i.e.\ exponential relaxation of \(v\) to \(V(\rho_0)\). This is why the source in [\((42)\)](#eq-arz-conservative2)
is chosen as \(\rho(V(\rho)-v)/\tau\).

The purpose of this local relaxation test is to isolate the intended role of the source term from the
transport dynamics. The PDE has two distinct mechanisms: (i) the hyperbolic part transports information through space,
and (ii) the source models drivers' finite-time adaptation of speed toward an equilibrium value.
By freezing \(\rho\) and dropping all \(x\)-derivatives we remove (i) and check that (ii) reduces to the correct
local behavior: at fixed density \(\rho_0\) a driver should relax exponentially to \(V(\rho_0)\) with time scale
\(\tau\). If the source does not pass this limit, then even spatially uniform traffic would have the wrong acceleration
law, so the model would be inconsistent. Passing the test is therefore a necessary consistency check: it confirms that
the chosen source implements the desired local driving ODE while leaving spatial (anisotropic) interaction to the
hyperbolic transport part.

From the balance law in \(y\) to the transported-marker form.

Now rewrite [\((42)\)](#eq-arz-conservative2) in terms of \(w=v+p(\rho)=y/\rho\). Compute

$$
\frac{\partial y}{\partial t}+\frac{\partial}{\partial x}(v y) =
\frac{\partial}{\partial t}(\rho w)+\frac{\partial}{\partial x}(\rho v w) =
\rho\Big(\frac{\partial w}{\partial t}+v\frac{\partial w}{\partial x}\Big) +
w\Big(\frac{\partial \rho}{\partial t}+\frac{\partial}{\partial x}(\rho v)\Big),
$$

where the last identity is just the product rule.
Using the continuity equation \(\frac{\partial \rho}{\partial t}+\frac{\partial}{\partial x}(\rho v)=0\), the last term vanishes, leaving

$$
\frac{\partial y}{\partial t}+\frac{\partial}{\partial x}(v y)=\rho\Big(\frac{\partial w}{\partial t}+v\frac{\partial w}{\partial x}\Big).
$$

But, from [\((42)\)](#eq-arz-conservative2) we have:

$$
\frac{\partial y}{\partial t}+\frac{\partial}{\partial x}(v y)=\rho \frac{V(\rho)-v}{\tau}.
$$

Combining the above two equations yields

<a id="eq-marker-relax-expanded-start"></a>

$$
\frac{\partial w}{\partial t}+v\,\frac{\partial w}{\partial x} = \frac{V(\rho)-v}{\tau}.
$$

**Comment.** Equation [\((49)\)](#eq-marker-relax-expanded-start) is the relaxation ARZ closure in its most transparent form:
the marker \(w=v+p(\rho)\) is still transported along vehicle trajectories (left-hand side),
so the anisotropic car-following structure is preserved, while the right-hand side introduces only a
local tendency that drives \(v\) toward the equilibrium speed \(V(\rho)\) on the time scale \(\tau\).
In particular, setting \(\tau\to\infty\) recovers the homogeneous convected-marker law, and if spatial gradients are
ignored the equation reduces to the desired ODE \(v_t=(V(\rho)-v)/\tau\) at fixed density.

### Connection to the LWR model

The first-order Lighthill--Whitham--Richards (LWR) model assumes an equilibrium relation \(v=V(\rho)\), reducing dynamics to a single conservation law

$$
\frac{\partial \rho}{\partial t} + \frac{\partial}{\partial x}\big(\rho V(\rho)\big)=0.
$$

In ARZ with relaxation, if the relaxation is fast (formally \(\tau\to 0^+\)), solutions are driven toward \(v\approx V(\rho)\),
and the density evolution approaches LWR with flux \(\rho V(\rho)\) (the equilibrium flux) [[3](#ref-3)].

## A common calibration choice: Greenshields-type equilibrium

One frequently used parametric form (appearing in ARZ control/estimation literature) sets, for \(\gamma>0\),

<a id="eq-pressure-greenshields"></a>

$$
p(\rho)=\rho^\gamma,
\qquad\text{or rescaled as}\qquad
p(\rho)=v_f\Big(\frac{\rho}{\rho_m}\Big)^\gamma,
$$

with free-flow speed \(v_f\) and maximum (jam) density \(\rho_m\) [[3](#ref-3)].
Then a consistent equilibrium speed can be written as

<a id="eq-v-greenshields"></a>

$$
V(\rho)=v_f-p(\rho) =
 v_f\Big(1-\Big(\frac{\rho}{\rho_m}\Big)^\gamma\Big),
$$

which is the (generalized) Greenshields-type relationship used in [[3](#ref-3)].

## Initial and boundary conditions (template)

For an initial-value problem on the line, specify

$$
\rho(x,0)=\rho_0(x),\qquad v(x,0)=v_0(x),
$$

with physically admissible constraints (e.g., \(\rho_0\ge 0\) and \(v_0\ge 0\)).
On a finite road segment \(x\in[0,L]\), appropriate boundary conditions depend on the traffic regime (which characteristic fields enter/leave the domain).
(We can add a detailed boundary-condition section later.)

## References

1. <a id="ref-1"></a>A.~Aw and M.~Rascle, "Resurrection of `Second Order' Models of Traffic Flow,"  SIAM Journal on Applied Mathematics, 60(3):916--938, 2000. DOI: 10.1137/S0036139997332099.

3. <a id="ref-2"></a>H.~M. Zhang, "A Non-Equilibrium Traffic Model Devoid of Gas-Like Behavior," Transportation Research Part B, 36(3):275--290, 2002.

4. <a id="ref-3"></a>H.~Yu and M.~Krstic, "Traffic Congestion Control for Aw--Rascle--Zhang Model," Automatica, 100:38--51, 2019.
   
6. <a id="ref-4"></a>M.~Garavello and S.~Villa, "The Cauchy Problem for the Aw--Rascle--Zhang Traffic Model with Locally Constrained Flow," 2016.
