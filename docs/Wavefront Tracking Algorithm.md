## Introduction

The Hughes model describes pedestrian motion by coupling density evolution with a dynamically determined walking direction. In one spatial dimension, the direction changes at a turning point that balances the travel cost toward the two exits. We consider a wave-front tracking approximation of this model, where the density is represented by piecewise-constant states separated by moving fronts. The method evolves these fronts through successive Riemann problems and interactions, providing a constructive approximation of the solution.

## The One-Dimensional Hughes Model

### Governing Equations

We consider pedestrians moving along the one-dimensional corridor \(\Omega=[-1,1]\), with exits located at the two endpoints \(x=-1\) and \(x=1\). The function \(\rho=\rho(x,t)\) denotes the pedestrian density at position \(x\in[-1,1]\) and time \(t\geq0\).

The one-dimensional Hughes model can be written as

$$
\rho_t-\partial_x\left(
\rho\,v(\rho)\frac{\phi_x}{|\phi_x|}
\right)=0,
$$

together with

$$
|\phi_x|=c(\rho).
$$

Here, \(\phi(x,t)\) represents the perceived cost of reaching an exit from
position \(x\) at time \(t\), \(v(\rho)\) denotes the walking speed at density
\(\rho\), and \(c(\rho)\) represents the travel cost per unit distance associated
with that density. In our case,

$$
v(\rho)=1-\rho,
\qquad
c(\rho)=\frac{1}{1-\rho},
$$

and therefore the scalar flux is

$$
f(\rho)=\rho v(\rho)=\rho(1-\rho).
$$

A central feature of the Hughes model is the turning point,
denoted by \(\xi(t)\). At each time \(t\), this point represents the
location at which the perceived cost of reaching the left exit is equal
to the perceived cost of reaching the right exit.

$$
\int_{-1}^{\xi(t)} c(\rho(x,t))\,dx =
\int_{\xi(t)}^{1} c(\rho(x,t))\,dx.
$$

Pedestrians located to the left of \(\xi(t)\) move toward the exit at
\(x=-1\), while pedestrians located to the right move toward the exit at
\(x=1\). Consequently, away from the turning point the conservation law takes
the form

$$
\rho_t-f(\rho)_x=0,
\qquad -1\lt x\lt\xi(t),
$$

on the left, and

$$
\rho_t+f(\rho)_x=0,
\qquad \xi(t)\lt x\lt 1,
$$

on the right.

### Initial and Boundary Data

Here the initial condition is assumed to be piecewise constant,

$$
\rho(x,0)=\rho_0(x),
\qquad x\in[-1,1].
$$

It is described by a finite set of jump locations

$$
-1\lt x_1\lt x_2\lt\cdots\lt x_{n_j}\lt 1,
$$

and one constant density value on each interval between two consecutive
jumps. The value
\(\rho=1\) is avoided, since the cost

$$
c(\rho)=\frac{1}{1-\rho}
$$

becomes singular at \(\rho=1\). At both exits we assume vacuum outside the corridor. Numerically, this is
implemented by adding zero-density states at \(x=-1\) and \(x=1\). These
states allow pedestrians to leave the domain through either exit.

## Wave-Front Tracking Approximation

We approximate the solution of the one-dimensional Hughes model using a
wave-front tracking (WFT) scheme. Given an initial pedestrian density
\(\rho_0(x)\) on the domain \([-1,1]\), the objective is to determine the
density field

$$
\rho(x,t), \qquad x\in[-1,1], \quad t\in[0,T].
$$

The WFT method approximates this solution by constant-density regions
separated by moving discontinuities, called wave fronts. These fronts are
generated from local Riemann problems and evolved in time until the final
time \(T\).

### Discretization of the Initial Density

Let \(\varepsilon>0\) denote the discretization parameter of the WFT scheme, with \(\varepsilon=1/N\) for some integer \(N\). To construct the wave-front tracking approximation of the initial density, we use the finite set:

$$
\mathcal{G}_{\varepsilon} =
\{0,\varepsilon,2\varepsilon,\ldots,1-\varepsilon\}.
$$

Each value taken by the initial density \(\rho_0\) at any \(x\), is then replaced by
the closest value in \(\mathcal{G}_{\varepsilon}\). Thus, the discretized initial density, denoted by \(\rho_0^\varepsilon\), remains piecewise constant and has the same jump locations as \(\rho_0\), while its values belong to \(\mathcal{G}_{\varepsilon}\). 

### Initial Turning Point

Once the initial density has been discretized, the initial turning point
\(\xi(0)\) is determined from the discrete cost balance. Let

$$
-1=x_0\lt x_1\lt\cdots\lt x_m=1
$$

denote the boundaries of the piecewise-constant density regions. These
points define \(m\) generally non-uniform intervals

$$
[x_{i-1},x_i), \qquad i=1,\ldots,m,
$$

where \(i\) denotes the interval index. The discretized density
\(\rho_i^\varepsilon\) is constant on each interval \([x_{i-1},x_i)\). To compute the initial turning point, we need the total cost at \(t=0\), given by:

$$
A_{\mathrm{tot}} =
\sum_{i=1}^{m}
(x_i-x_{i-1})\,c(\rho_i^\varepsilon).
$$

The turning point \(\xi(0)\) is determined by comparing the accumulated
cost from the left with half of the total cost,

$$
\frac{A_{\mathrm{tot}}}{2}.
$$

Suppose this value is reached between two consecutive interval boundaries,
\(x_{k-1}\) and \(x_k\). Then the turning point lies inside that interval,

$$
\xi(0)\in[x_{k-1},x_k].
$$

Since the density is constant on this interval, the cost increases
linearly with distance. Therefore, \(\xi(0)\) is found by moving from
\(x_{k-1}\) just far enough for the accumulated cost to become exactly
\(A_{\mathrm{tot}}/2\). If the half-cost value is reached exactly at
\(x_k\), then \(\xi(0)=x_k\).

### Summary of the Initial Setup

At this point, the initial density \(\rho_0^\varepsilon\) has been
discretized according to \(\rho_i^\varepsilon\), and the initial turning point \(\xi(0)\) has been determined
from the cost balance. The turning point separates the left-moving and right-moving
regions. These quantities provide the complete initial data needed to
construct the Riemann problems and generate the wave fronts.

### Riemann Problems

A Riemann problem is a local conservation-law problem with two constant
states separated by a single discontinuity. For example, at a jump
location \(x=x_j\),

$$
\rho(x,0)=
\begin{cases}
\rho_L, & x<x_j,\\
\rho_R, & x>x_j.
\end{cases}
$$

The purpose is to determine how this jump evolves: either as a single
shock or as a rarefaction wave. In the WFT method, a rarefaction wave between \(\rho_L\) and \(\rho_R\) is
replaced by a sequence of smaller density jumps, where two consecutive
density values differ by \(\varepsilon\). Each
small jump is treated as an individual shock and these shocks are propagated with
their corresponding Rankine-Hugoniot speed. Together, these fronts form
a discrete approximation of the rarefaction wave. Note that, \(\xi(t)\) is treated as
a moving front with its own propagation speed as well, the cost calculation is used to determine the turning-point
position only at \(t=0\)

#### Riemann Problems Away from the Turning Point

For each initial jump away from the turning point, a Riemann problem is
solved independently. On the left of \(\xi\), where the flux is \(-f(\rho)\),
a jump is kept as a shock when \(\rho_L>\rho_R\), otherwise it is replaced
by a rarefaction fan. On the right of \(\xi\), where
the flux is \(+f(\rho)\), the corresponding shock/rarefaction condition is
reversed.

Each resulting front is assigned a propagation speed using the
Rankine--Hugoniot relation. Thus, this step converts every initial density
jump into a finite set of moving wave fronts. At the exits, any front located at \(x=-1\) and moving out of the domain is
removed, similarly, any front at \(x=1\) moving outward is discarded. A
stationary boundary front is then retained at each exit.

#### Riemann Problem at the Turning Point

The Riemann problem at \(\xi(t)\) is different from the problems considered
away from the turning point. On the left, pedestrians satisfy the
conservation law with flux \(-f(\rho)\), while on the right they satisfy
the conservation law with flux \(+f(\rho)\). Moreover, the position
\(\xi(t)\) is not prescribed: it must move so that the travel cost toward
the two exits remains balanced.

If \(t_k\) and \(t_{k+1}\) are two consecutive front collision times, the
turning point is propagated together with the other wave fronts on the
interval \(t\in[t_k,t_{k+1})\). During this entire propagation interval,
the balance condition

$$
\Psi_L(t)=\Psi_R(t)
$$

must remain satisfied, where

$$
\Psi_L(t)=\int_{-1}^{\xi(t)}c(\rho(x,t))\,dx,
\qquad
\Psi_R(t)=\int_{\xi(t)}^{1}c(\rho(x,t))\,dx.
$$

Therefore, the turning-point solver must determine not only the waves
generated near \(\xi(t_k)\), but also the state and propagation speed at
the turning point that preserve this balance until the next interaction.

Effect of the current wave configuration.

Since each constant-density interval is bounded by moving wave fronts,
the length of these intervals changes as the fronts move. This changes
their contribution to the total travel cost over time as well.

To account for the motion of the entire current wave configuration, the
solver considers all fronts on both sides of the turning point: from
\(x=-1\) to \(\xi(t)\) on the left, and from \(\xi(t)\) to \(x=1\) on the right.
It then computes

$$
\psi^\ast =
\sum_i s_i^R
\bigl[c(\rho_i^R)-c(\rho_{i+1}^R)\bigr] -
\sum_i s_i^L
\bigl[c(\rho_i^L)-c(\rho_{i+1}^L)\bigr].
$$

Here, \(s_i^L\) and \(s_i^R\) denote the propagation speeds of the individual
wave fronts on the left and right of \(\xi(t)\), respectively, while
\(\rho_i\) and \(\rho_{i+1}\) are the constant density states separated by
that front. The index \(i\) therefore runs over all fronts currently present
on the corresponding side.

This should not be confused with collision detection. Collision detection
only compares neighbouring fronts to find which interaction occurs next.
In contrast, \(\psi^\ast\) uses the entire current wave configuration and
describes how the motion of these fronts is changing the balance between
the left and right travel costs.

Choosing the state at the turning point.

At a given interaction time, the position of the turning point \(\xi(t)\)
is already known from its previous propagation. What must now be updated
is the local wave configuration around \(\xi(t)\) and the speed with which
the turning point will move during the next time interval. The solver starts from three quantities,

$$
\rho_L,\qquad \rho_R,\qquad \psi^\ast,
$$

where

$$
\rho_L=\rho(\xi(t)^-,t),
\qquad
\rho_R=\rho(\xi(t)^+,t)
$$

are the density states immediately to the left and right of the turning
point. These are local quantities: they describe only the two states
touching \(\xi(t)\). In contrast, \(\psi^\ast\) is a global quantity obtained
from all wave fronts currently present on both sides of the domain. It
describes how the motion of the current fronts affects the left--right
travel-cost balance.

The purpose of the turning-point solver is therefore to find a new
intermediate state, denoted by \(\rho_m\), and a new turning-point speed
\(s_m\) such that the cost balance remains satisfied during the next period
of propagation. The first distinction is made according to the two neighbouring states,

$$
\rho_L=\rho_R,\qquad
\rho_L>\rho_R,\qquad\text{or}\qquad
\rho_L<\rho_R.
$$

These three cases lead to different possible wave patterns around the
turning point: a single connecting front, a discretized rarefaction fan,
or a vacuum region with \(\rho_m=0\). The particular pattern depends on
the relation between \(\rho_L\) and \(\rho_R\) and on the current
cost-balance quantity \(\psi^\ast\). To determine which pattern is admissible, the value of
\(\psi^\ast\) is compared with threshold values obtained from four auxiliary
functions,

$$
\theta,\qquad \lambda,\qquad \xi_1,\qquad \xi_2.
$$

The functions \(\theta\) and \(\lambda\) correspond to the two possible
single-front configurations around the turning point. For the \(\theta\) configuration, the turning-point front separates
\(\rho_L\) and \(\rho_m\), while the additional connection
\(\rho_m\to\rho_R\) lies to the right of \(\xi(t)\). For the \(\lambda\) configuration, the turning-point front separates
\(\rho_m\) and \(\rho_R\), while the additional connection
\(\rho_L\to\rho_m\) lies to the left of \(\xi(t)\). Thus, \(\theta\) and \(\lambda\) describe two different ways of inserting the
intermediate state \(\rho_m\) between the neighbouring states
\(\rho_L\) and \(\rho_R\) while satisfying the cost-balance condition.

For the \(\xi_1\) configuration, the front located at the turning point
separates \(\rho_L\) and the intermediate state \(\rho_m\), while the
connection from \(\rho_m\) to \(\rho_R\) is resolved as a rarefaction fan
on the right of \(\xi(t)\). For the \(\xi_2\) configuration, the arrangement is reversed: the
turning-point front separates \(\rho_m\) and \(\rho_R\), while the
connection from \(\rho_L\) to \(\rho_m\) is resolved as a rarefaction fan
on the left of \(\xi(t)\). Here, \(\xi_1\) and \(\xi_2\) are auxiliary scalar functions and should not
be confused with the turning-point position \(\xi(t)\).

Once the appropriate case is identified,  the
corresponding equation

$$
\theta(\rho_m)=\psi^\ast,\qquad
\lambda(\rho_m)=\psi^\ast,\qquad
\xi_1(\rho_m)=\psi^\ast,
\quad\text{or}\quad
\xi_2(\rho_m)=\psi^\ast
$$

is solved for \(\rho_m\). Only one of these equations is solved for a given
configuration. 

#### Construction of the outgoing fronts.

Once the appropriate case has been selected, the intermediate state
\(\rho_m\) is projected onto the \(\varepsilon\)-grid. The front located
exactly at the turning point connects \(\rho_m\) to one of the neighbouring
states, either \(\rho_L\) or \(\rho_R\), and its propagation speed \(s_m\) is
computed from the corresponding Rankine--Hugoniot relation.

The intermediate state \(\rho_m\) must then also be connected to the state
on the opposite side of the turning point. For example, if the
turning-point front connects \(\rho_L\) to \(\rho_m\), then the second
connection must join \(\rho_m\) to \(\rho_R\). Conversely, if the
turning-point front connects \(\rho_m\) to \(\rho_R\), then the second
connection must join \(\rho_L\) to \(\rho_m\). This second connection is constructed according to the selected wave
pattern. If it is a shock, a single additional front is introduced. If
it is a rarefaction, it is represented by an \(\varepsilon\)-spaced fan of
fronts, with each front assigned its own propagation speed. In the vacuum case,

$$
\rho_m=0,
$$

and the turning-point speed is obtained directly from the cost-balance
condition rather than from a Rankine--Hugoniot ratio.

This completes the updated local configuration at \(\xi(t)\), which is then
propagated together with the existing fronts until the next interaction.

## Time evolution

Once the fronts and their speeds are known, all wave fronts,
including the turning point \(\xi(t)\), are propagated at constant speed
until the next interaction occurs. At that time, the wave configuration
is updated and new propagation speeds are determined.

### Front Propagation

Between two consecutive interactions, each front moves linearly. If a
front is located at \(x_j\) with speed \(s_j\) at time \(t\), then its position
after a time increment \(\tau\) is

$$
x_j(t+\tau)=x_j(t)+s_j\,\tau.
$$

The turning point is propagated in the same way using its speed \(s_m\).
Thus, all fronts keep constant speeds until a collision.

### Collision Detection

For two neighbouring fronts at positions \(x_j<x_{j+1}\) with speeds
\(s_j\) and \(s_{j+1}\), their possible collision time is

$$
\tau_j=
\frac{x_{j+1}-x_j}{s_j-s_{j+1}}.
$$

Only positive values of \(\tau_j\) correspond to future collisions. The
next interaction time is therefore

$$
\tau_c=\min_{\tau_j>0}\tau_j.
$$

If no positive collision time exists, or if the next collision occurs
after the final time \(T\), the current fronts are simply propagated to
\(T\). Otherwise, all fronts are advanced by \(\tau_c\), the colliding
fronts are merged, and the corresponding Riemann problem is solved again
to determine the new states and front speeds. 

### Front Interaction and Re-Solving

When two or more fronts collide, the corresponding discontinuities are
merged and the local wave configuration is updated. The density states
on both sides of the interaction are retained as the new Riemann data. The Riemann solver is then applied again to this updated configuration.
Any new jump is resolved into either a shock front or a discretized
rarefaction fan, and the turning-point problem is re-solved using the
new neighbouring states.

This produces a new set of front positions, density states, and
propagation speeds, which are then used until the next collision occurs.
The procedure is repeated until the final time \(T\) is reached.

## Reconstruction of the Solution to a grid

The WFT solution is stored through the front positions, their speeds, and
the constant density states between consecutive interaction times. To
obtain the solution on a fixed space--time grid, these fronts are
reconstructed at the desired sampling points. We introduce the uniform grids

$$
x_i=-1+i\Delta x,
\qquad
\Delta x=\frac{2}{N_x-1},
$$

and

$$
t_n=n\Delta t,
\qquad
\Delta t=\frac{T}{N_t-1},
$$

where \(N_x\) and \(N_t\) are the numbers of spatial and temporal grid
points. For each time \(t_n\), the corresponding interaction interval is first
identified. The stored fronts are then propagated from the last
interaction time to \(t_n\) using their constant speeds,

$$
x_j(t_n)=x_j(t_k)+s_j(t_n-t_k).
$$

At an interaction time, two or more fronts may occupy the same numerical
position. In this case, only the last coincident front is retained, so
that the outgoing density state after the collision is used in the
reconstruction.

The density is piecewise constant between these reconstructed front
positions. Therefore, each spatial grid point \(x_i\) is assigned the
density of the interval containing it. Repeating this procedure for all
\(t_n\) produces the numerical solution

$$
\rho_{n,i}\approx\rho(x_i,t_n).
$$

## References

1. P. Goatin and M. Mimault,
The wave-front tracking algorithm for Hughes' model of pedestrian motion,
SIAM Journal on Scientific Computing,
vol. 35, no. 3, pp. B606--B622, 2013.

2. P. Chauhan, S. E. Choutri, M. Ghattassi, N. Masmoudi, and S. E. Jabari,
Neural operators struggle to learn complex PDEs in pedestrian mobility:
Hughes model case study,
Artificial Intelligence for Transportation,
vol. 1, Art. no. 100005, 2025.
