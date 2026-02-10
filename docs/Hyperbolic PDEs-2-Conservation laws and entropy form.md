## Introduction to conservation law

A conservation law is a mathematical statement that something is neither created nor destroyed inside a region, except by flowing across the region's boundary or by external sources. The key idea is change = stuff in - stuff out + sources. Suppose, \(u(x,t)\) represent the density of that conserved quantity at position \(x\) and time \(t\). Then the conservation principle in 1D says:

$$
\frac{d}{dt} \Big(\text{amount of $u$ in an interval}\Big) =
\text{(flux entering)} - \text{(flux leaving)} + \text{(sources inside)}.
$$

This flowing in and flowing out of any quantity is defined by the term flux. Flux is the rate at which the conserved quantity crosses a boundary. In 1D, the boundary of an interval \([a,b]\) is just the two endpoints \(a\) and \(b\). If \(f(u)\) is the flux function, then:

- \(f(u(a,t))\) is the rate at which \(u\) flows from left to right across \(x=a\),
- \(f(u(b,t))\) is the rate at which \(u\) flows from left to right across \(x=b\).

### Conservation law in integral and differential form

The amount of density \((u)\) in the interval \([a,b]\) is given by:

$$
\int_a^b u(x,t)dx.
$$

If the flux in the positive \(x-\)direction is \(f(u)\), assuming no sources for now, gives the conservation law as:

$$
\frac{d}{dt}\int_a^b u(x,t)dx = f(u(a,t)) - f(u(b,t)) + 0.
$$

This is the conservation law in integral form. It is extremely important because it remains meaningful even when \(u\) is not smooth (for example, if there is a shock).

### From integral form to differential form

Assume \(u\) is smooth enough that we can move derivatives inside integrals and use the fundamental theorem of calculus. We can rewrite the above as:

$$
\frac{d}{dt}\int_a^b udx + \big(f(u(b,t)) - f(u(a,t))\big) = 0.
$$

Notice that:

$$
\frac{d}{dt}\int_a^b u(x,t)dx
= \int_a^b \frac{\partial u}{\partial t}(x,t)dx.
$$

and

$$
f(u(b,t)) - f(u(a,t)) = \int_a^b \frac{\partial}{\partial x}f(u(x,t))dx.
$$

So:

$$
\int_a^b \left( \frac{\partial u}{\partial t}(x,t) + \frac{\partial}{\partial x}f(u(x,t))\right)dx = 0.
$$

If this holds for every interval \([a,b]\), then in the global sense, the integrand must be zero:

$$
\frac{\partial u}{\partial t}(x,t) + \frac{\partial}{\partial x}f(u(x,t)) = u_t + \big(f(u)\big)_x = 0.
$$

This is the differential form of a 1D conservation law.

### Adding sources

If there is creation or removal inside the region, represented by a source term \(s(x,t)\), then:

$$
\frac{d}{dt}\int_a^b udx = f(u(a,t)) - f(u(b,t)) + \int_a^b s(x,t)dx,
$$

and the differential form becomes:

$$
u_t + (f(u))_x = s(x,t).
$$

## Finite Volume Method (FVM)

The finite volume method is built directly from the integral conservation law, so it automatically respects conservation. It works by integrating the PDE over small control volumes (cells) and approximating the fluxes crossing the cell boundaries. Information is exchanged between cells only through these fluxes, so a conservative flux approximation immediately yields a globally conservative scheme.

Step 1: break space into cells:

For any given cell \(i\), with center \(x_i\). Divide the 1D space into cells:

$$
[x_{i-\tfrac12}, x_{i+\tfrac12}], \quad \Delta x = x_{i+\tfrac12}-x_{i-\tfrac12}.
$$

Then we can define the cell average (per unit length for 1D case):

$$
\bar{u}_i(t) = \frac{1}{\Delta x}\int_{x_{i-\tfrac12}}^{x_{i+\tfrac12}} u(x,t)dx.
$$

Step 2: integrate the PDE over a cell:

Start from the differential form:

$$
u_t + (f(u))_x = 0.
$$

Integrate over cell \(i\):

$$
\int_{x_{i-\tfrac12}}^{x_{i+\tfrac12}} u_tdx + 
\int_{x_{i-\tfrac12}}^{x_{i+\tfrac12}} (f(u))_xdx = 0.
$$

From the first term, we can say for a given cell:

$$
\int u_tdx = \frac{d}{dt}\int udx = \frac{d}{dt}\big(\Delta x\bar{u}_i(t)\big).
$$

The second term uses the fundamental theorem of calculus:

$$
\int (f(u))_xdx = f(u(x_{i+\tfrac12},t)) - f(u(x_{i-\tfrac12},t)).
$$

So:

$$
\frac{d}{dt}\big(\Delta x\bar{u}_i\big) +
 \big(f_{i+\tfrac12} - f_{i-\tfrac12}\big)=0,
$$

where \(f_{i\pm \tfrac12}\) denotes the flux through the cell face. Dividing by \(\Delta x\) gives:

$$
\frac{d\bar{u}_i}{dt} = -\frac{1}{\Delta x}\left(f_{i+\tfrac12}-f_{i-\tfrac12}\right).
$$

### Key numerical problem

The above equation shows that the evolution of \(\bar{u}_i\) is determined by the boundary or face fluxes \(f_{i\pm \tfrac12}\), but these are not directly known from the stored data. We do not know the true face fluxes \(f_{i\pm \tfrac12}\) because we only store
cell averages. Instead, we introduce a numerical flux at each interface:

$$
\hat{f}_{i+\tfrac12} = \hat{f}\big(u_{i+\tfrac12}^L,u_{i+\tfrac12}^R\big), 
\qquad
\hat{f}_{i-\tfrac12} = \hat{f}\big(u_{i-\tfrac12}^L,u_{i-\tfrac12}^R\big),
$$

where \(u_{i\pm\tfrac12}^L\) and \(u_{i\pm\tfrac12}^R\) are the reconstructed states
immediately to the left and right of the interface \(x_{i\pm\tfrac12}\), obtained from
the neighboring cells. For example:

- at \(x_{i+\tfrac12}\), \(u_{i+\tfrac12}^L\) comes from cell \(i\) and \(u_{i+\tfrac12}^R\) from cell \(i+1\),
- at \(x_{i-\tfrac12}\), \(u_{i-\tfrac12}^L\) comes from cell \(i-1\) and \(u_{i-\tfrac12}^R\) from cell \(i\).

The semi-discrete finite volume update is then

$$
\frac{d\bar{u}_i}{dt}
  = -\frac{1}{\Delta x}\left(\hat{f}_{i+\tfrac12}-\hat{f}_{i-\tfrac12}\right).
$$

### Time stepping (turning \(d\bar{u}/dt\) into updates)

The semi-discrete scheme given above gives, for each cell, an ODE in time for the averages \(\bar{u}_i(t)\),
where the right-hand side depends on the cell averages at the current time but not explicitly on \(x\). Thus, the spatial dependence has been replaced by cell indices, leaving a system of ODEs in the single variable \(t\). To obtain an implementable method, we now discretize time by choosing a time-stepping scheme for this system. A common simplest method is Forward Euler in time:

$$
\bar{u}_i^{n+1} = \bar{u}_i^n - \frac{\Delta t}{\Delta x}\left(\hat{f}_{i+\tfrac12}^n-\hat{f}_{i-\tfrac12}^n\right),
$$

where \(n\) represents the \(n^{th}\) time step for any cell \(i\). Higher accuracy in time uses methods like Runge--Kutta (RK2, RK3), but the flux idea stays the same.

### CFL condition

Information in hyperbolic conservation laws travels at finite speeds (wave speeds).
Numerically, we do not want a wave to travel farther than one cell in a single
time step: the fastest signal should not skip over cells, so that each update depends only on information from neighboring cells. If \(|\lambda|\) is a
characteristic wave speed, then in one time step the distance traveled is
approximately \(|\lambda|\Delta t\). To keep this below one cell width \(\Delta x\), we
require

$$
|\lambda|\Delta t \;\lesssim\; \Delta x
\quad\Longrightarrow\quad
\Delta t \;\le\; \text{CFL}\cdot \frac{\Delta x}{\max |\lambda|},
$$

where \(\text{CFL}\in(0,1]\) is a chosen safety factor. For our scalar conservation law \(u_t + (f(u))_x = 0\), the characteristic speed is

$$
\lambda = f'(u),
$$

so we typically use \(\max |\lambda| = \max |f'(u)|\) over the grid when computing
the time step.

## From interface states to numerical fluxes

At each cell interface the solution is, in general, discontinuous, so the left and
right traces differ:

$$
u(x_{i+\tfrac12}^-,t) \approx u_L, \qquad
u(x_{i+\tfrac12}^+,t) \approx u_R.
$$

This setup defines a local Riemann problem: the PDE with piecewise constant
initial data having a single jump at \(x_{i+\tfrac12}\). A good numerical flux
\(\hat{f}(u_L,u_R)\) aims to approximate the physical flux at the interface
generated by the solution of this local Riemann problem. As an example, Godunov's method can be used to approximate this flux.

### Godunov's idea

1. Approximate the solution in each cell by a constant (piecewise constant reconstruction).
2. At every interface, pose and solve the corresponding Riemann problem exactly.
3. Take the interfacial flux from this exact Riemann solution as the numerical flux.

This procedure is very faithful to the underlying physics, but exact Riemann solvers
are often costly or complicated. In practice, we therefore use
approximate Riemann solvers, i.e.\ closed-form numerical flux formulas
that mimic the behavior of the exact solution. For Godunov's original scheme we use a first-order (piecewise constant) reconstruction:

$$
u_L = \bar{u}_i, \qquad u_R = \bar{u}_{i+1},
$$

so the numerical flux at \(x_{i+\tfrac12}\) reduces to a function
\(\hat{f}(\bar{u}_i,\bar{u}_{i+1})\) of neighboring cell averages only. This is very robust,
but the low-order reconstruction makes the method diffusive, smearing sharp
discontinuities and steep gradients.

### Higher order reconstruction

To reduce this numerical diffusion, we can move to a higher-order (piecewise linear)
reconstruction. In each cell we reconstruct a slope \(\sigma_i\) and define

$$
u_{i+\tfrac12}^L = \bar{u}_i + \frac{1}{2}\sigma_i, \qquad
u_{i+\tfrac12}^R = \bar{u}_{i+1} - \frac{1}{2}\sigma_{i+1},
$$

so that the left and right states at the interface \(x_{i+\tfrac12}\) come from linear
profiles inside cells \(i\) and \(i+1\). The slopes \(\sigma_i\) are typically chosen with a
limiter (e.g.\ minmod, van Leer, MC) to obtain second-order accuracy in smooth
regions. A limiter is a nonlinear function that reduces the slope when neighboring
cell averages are inconsistent (e.g.\ near a discontinuity), preventing the creation
of new spurious extrema and suppressing Gibbs-type oscillations.

Near true discontinuities the limiter forces the reconstruction to drop back to
(first-order) more diffusive behavior, which is unavoidable if we want to avoid
spurious oscillations. The gain is that away from shocks the scheme remains
second-order accurate, so smooth regions are much less smeared than with a
purely first-order method. In summary, a high-resolution finite volume scheme
consists of a reconstruction procedure plus a suitable numerical flux
\(\hat{f}(u_L,u_R)\). There are several popular numerical fluxes for scalar conservation laws. Each
offers a different trade-off between accuracy, robustness, and computational cost. A few example of those are upwind flux, Lax-Friedrichs (Rusanov) flux, Godunov flux, HLL-type flux and many more.

## Entropy-conservative and entropy-stable fluxes

For many conservation laws it is useful to control, in addition to the
conserved variables \(u\), a convex entropy \(\eta(u)\).
For the scalar conservation law

$$
u_t + f(u)_x = 0,
$$

an entropy/entropy-flux pair \((\eta,q)\) is defined by the relation

$$
q'(u) = \eta'(u)f'(u).
$$

Assuming \(u\) is smooth, the chain rule gives

$$
\eta(u)_t = \eta'(u)u_t,
\qquad
q(u)_x = q'(u)u_x.
$$

From our conservation law we have \(u_t = -f'(u)u_x\), hence

$$
\eta(u)_t
  = \eta'(u)u_t
  = -\eta'(u)f'(u)u_x
  = -q'(u)u_x
  = -q(u)_x.
$$

Therefore we get a conservation law in entropy terms as

$$
\eta(u)_t + q(u)_x = 0
$$

for smooth solutions, and the inequality
\(\eta(u)_t + q(u)_x \le 0\) for weak (entropy) solutions. This is discussed in great detail in chapter 2 (weak solutions) of this series.

### Entropy variables and flux potential

If we extend our analysis to systems (more than one conserved quantity), we can write \(u\) as a vector. We still have our conservation law for smooth solutions in entropy terms as

$$
\eta(u)_t + q(u)_x = 0
$$

Then, we can define the entropy variables \(v := \eta'(u)\). Using the chain rule, the above equation can be written as

$$
v^\top u_t + q'(u) u_x = 0.
$$

From our conservation law \(u_t = -f(u)_x = -f'(u)u_x\) we obtain

$$
-v^\top f'(u)u_x + q'(u)u_x = 0
\quad\Rightarrow\quad
\bigl(q'(u) - v^\top f'(u)\bigr)u_x = 0.
$$

Since this must hold for arbitrary smooth \(u(x,t)\), we require the
compatibility condition

$$
q'(u) = v^\top f'(u)
\quad\text{or}\quad
\nabla_u q(u) = \eta'(u)^\top f'(u).
$$

Using the chain rule and the compatibility relation
from above, we have

$$
q(u)_x = q'(u)u_x = v^\top f'(u)u_x = v^\top f(u)_x.
$$

Thus the entropy law can already be written as

$$
\eta(u)_t + v^\top f(u)_x = 0.
$$

**Interface notation.**

At each cell interface \(x_{i+\frac12}\) we have a left state and a
right state.  For example,
\(u_{i+\frac12}^-\) and \(u_{i+\frac12}^+\) are the left and right limits
of \(u(x)\) at \(x_{i+\frac12}\), and the corresponding entropy variables
are \(v_{i+\frac12}^\pm := \eta'(u_{i+\frac12}^\pm)\).

### Entropy-conservative numerical flux

In a finite-volume method, the only information available at the interface
\(x_{i+\frac12}\) is the left and right cell states
\(u_{i+\frac12}^-\) and \(u_{i+\frac12}^+\).  
Therefore, we use a two-point numerical flux, which is simply a function that uses only these two states to approximate the physical flux at the interface:

$$
f^{\mathrm{ec}}_{i+\frac12}
  = f^{\mathrm{ec}}(u_{i+\frac12}^-,u_{i+\frac12}^+),
$$

with the consistency requirement
\(f^{\mathrm{ec}}(u,u) = f(u)\) where \(ec\) denotes  entropy-conservative.

We now ask for this flux to be compatible with the entropy structure. Consider the semi-discrete finite-volume scheme built from
\(f^{\mathrm{ec}}_{i+\frac12}\),

$$
\frac{d}{dt}\bar{u}_i(t)
  = -\frac{1}{\Delta x}
    \bigl(f^{\mathrm{ec}}_{i+\frac12} - f^{\mathrm{ec}}_{i-\frac12}\bigr).
$$

Using the chain rule cell by cell, we first differentiate the entropy
in each cell:

$$
\frac{d}{dt}\eta(\bar u_i(t))
  = \eta'(\bar u_i(t))^\top \frac{d}{dt}\bar u_i(t)
  = v_i^\top \frac{d}{dt}\bar u_i(t),
$$

since by definition \(v_i := \eta'(\bar u_i)\). Now multiply by the cell width \(\Delta x\) and sum over all cells (substituting value from the semi discrete finite volume scheme):

$$
\sum_i v_i^\top \frac{d}{dt}\bar u_i(t)\Delta x = -\sum_i v_i^\top
       \bigl(f^{\mathrm{ec}}_{i+\frac12}-f^{\mathrm{ec}}_{i-\frac12}\bigr).
$$

A discrete integration-by-parts is just an index shift that makes the
interface contributions telescope in the same way as in the continuous
case. For this we start from

$$
-\sum_i v_i^\top
   \bigl(f^{\mathrm{ec}}_{i+\frac12}-f^{\mathrm{ec}}_{i-\frac12}\bigr)
  = -\sum_i v_i^\top f^{\mathrm{ec}}_{i+\frac12}
    + \sum_i v_i^\top f^{\mathrm{ec}}_{i-\frac12}.
$$

In the second sum, perform the index shift \(j = i-1\):

$$
\sum_i v_i^\top f^{\mathrm{ec}}_{i-\frac12}
  = \sum_j v_{j+1}^\top f^{\mathrm{ec}}_{j+\frac12},
$$

so, renaming \(j\) back to \(i\),

$$
-\sum_i v_i^\top
   \bigl(f^{\mathrm{ec}}_{i+\frac12}-f^{\mathrm{ec}}_{i-\frac12}\bigr)
  = -\sum_i v_i^\top f^{\mathrm{ec}}_{i+\frac12}
    + \sum_i v_{i+1}^\top f^{\mathrm{ec}}_{i+\frac12}.
$$

Now combine the two sums:

$$
-\sum_i v_i^\top
   \bigl(f^{\mathrm{ec}}_{i+\frac12}-f^{\mathrm{ec}}_{i-\frac12}\bigr)
  = \sum_i (v_{i+1} - v_i)^\top f^{\mathrm{ec}}_{i+\frac12}.
$$

If we interpret the left cell of interface \(x_{i+\frac12}\) as
\(v_{i+\frac12}^- := v_i\) and the right cell as
\(v_{i+\frac12}^+ := v_{i+1}\), then

$$
[v]_{i+\frac12}
  := v_{i+\frac12}^+ - v_{i+\frac12}^-
   = v_{i+1} - v_i,
$$

and we can write

$$
-\sum_i v_i^\top
   \bigl(f^{\mathrm{ec}}_{i+\frac12}-f^{\mathrm{ec}}_{i-\frac12}\bigr) =
   \sum_i [v]_{i+\frac12}^\top f^{\mathrm{ec}}_{i+\frac12}.
$$

This is the discrete analogue of an integration-by-parts formula. To make this look like a discrete conservation law for the entropy case, we
require that each interface contribution can be written as a jump
of a scalar potential \(\psi\), i.e.

$$
[v]_{i+\frac12}^\top f^{\mathrm{ec}}_{i+\frac12} =
   [\psi]_{i+\frac12} :=
   \psi\bigl(v_{i+\frac12}^+\bigr) -
      \psi\bigl(v_{i+\frac12}^-\bigr)
$$

for all left/right states.  This is the discrete entropy identity. With this condition, the total discrete entropy satisfies

$$
\frac{d}{dt}\sum_i \eta(\bar u_i)\Delta x =
\sum_i [\psi]_{i+\frac12},
$$

which telescopes and vanishes under periodic (or suitable) boundary
conditions. This is because each term \([\psi]_{i+\frac12} = \psi_{i+\frac12}^+ - \psi_{i+\frac12}^-\) is a
difference of neighboring interface values, so

$$
\sum_i [\psi]_{i+\frac12}=
\sum_i \bigl(\psi_{i+\frac12}^+ - \psi_{i+\frac12}^-\bigr)
$$

forms a telescoping sum: all interior contributions cancel pairwise, leaving
only boundary terms.  For periodic boundaries (or if the entropy flux is
zero at the physical boundaries), these remaining boundary terms also cancel
(or vanish), so the total sum is zero and the discrete entropy is conserved.

$$
\frac{d}{dt}\sum_i \eta(\bar u_i)\Delta x = 0.
$$

### Adding dissipation: entropy-stable fluxes

Exact entropy conservation is too weak in the presence of shocks: the
physically relevant (entropy) solution satisfies
\(\eta(u)_t + q(u)_x \le 0\), i.e.\ entropy should decrease, not be
preserved.  Our entropy-conservative flux \(f^{\mathrm{ec}}_{i+\frac12}\)
gives

$$
\frac{d}{dt}\sum_i \eta(\bar u_i)\Delta x = 0,
$$

so we now modify the interface fluxes in such a way that the total
entropy becomes nonincreasing.

The idea is to add a dissipative term at each interface that is
negative definite in terms of the entropy variables.  Since
\(v = \eta'(u)\) are the natural variables for the entropy, we take a
correction that is linear in the jump \([v]_{i+\frac12}\) and symmetric
between the two neighboring cells.  This leads to the ansatz

$$
\hat f_{i+\frac12}
  := f^{\mathrm{ec}}_{i+\frac12}
     - \tfrac12 D_{i+\frac12}[v]_{i+\frac12},
$$

where \(D_{i+\frac12}\) is a symmetric positive semidefinite matrix
(\(D_{i+\frac12} \ge 0\) in the scalar case).

The form \(f^{\mathrm{ec}} - \tfrac12 D[v]\) is chosen for two reasons:

- The correction is centered at the interface and depends only on
  the jump \([v]_{i+\frac12} = v_{i+\frac12}^+ - v_{i+\frac12}^-\),
  so it vanishes for smooth (locally constant) states and does not
  destroy conservation of \(u\).
- When we repeat the entropy calculation with \(\hat f_{i+\frac12}\),
  the extra term contributes

$$
-\frac12\sum_i [v]_{i+\frac12}^\top
                D_{i+\frac12}[v]_{i+\frac12} \le 0
$$

  to \(\dfrac{d}{dt}\sum_i \eta(\bar u_i)\Delta x\), because
  \(D_{i+\frac12}\) is positive semidefinite.  Thus the total
  discrete entropy becomes nonincreasing, which is exactly the
  discrete analogue of the continuous entropy inequality.

In this sense, an entropy-stable flux is obtained by starting from an
entropy-conservative flux and adding a carefully designed, symmetric
dissipation term written in entropy variables.

Using \(\hat f_{i+\frac12}\) in our semi-discrete finite volume method, yields the
semi-discrete scheme

$$
\frac{d}{dt}\bar{u}_i(t)
  = -\frac{1}{\Delta x}
    \bigl(\hat f_{i+\frac12} - \hat f_{i-\frac12}\bigr).
$$

Repeating the above summation argument,

$$
\begin{aligned}
\frac{d}{dt}\sum_i \eta(\bar u_i)\Delta x
  &= \sum_i [v]_{i+\frac12}^\top \hat f_{i+\frac12} \\
  &= \sum_i [v]_{i+\frac12}^\top f^{\mathrm{ec}}_{i+\frac12}
     - \frac12\sum_i [v]_{i+\frac12}^\top
                      D_{i+\frac12}[v]_{i+\frac12}.
\end{aligned}
$$

The first term is the same as before and equals
\(\sum_i [\psi]_{i+\frac12}\), which vanishes under periodic (or suitable)
boundary conditions.  Thus

$$
\frac{d}{dt}\sum_i \eta(\bar u_i)\Delta x
  = -\frac12\sum_i [v]_{i+\frac12}^\top
                    D_{i+\frac12}[v]_{i+\frac12}
  \;\le\; 0,
$$

because each quadratic form
\([v]_{i+\frac12}^\top D_{i+\frac12}[v]_{i+\frac12}\) is nonnegative.
Hence the total discrete entropy is nonincreasing in time, and the scheme
is called entropy-stable.

In practice, constructing an entropy-stable finite-volume scheme proceeds
in two steps:

1. Design an entropy-conservative flux \(f^{\mathrm{ec}}_{i+\frac12}\)
   satisfying \eqref{eq:ec_condition};
2. Add symmetric dissipation in the form \eqref{eq:es_flux} with a
   suitable positive semidefinite matrix \(D_{i+\frac12}\)
   (often related to the absolute value of the Jacobian \(f'(u)\)).

The resulting method preserves the conservative structure of
\eqref{eq:fv_numflux} while enforcing a discrete entropy inequality,
providing nonlinear stability for the numerical solution.

### Entropy stable flux formulation

In the previous subsection, we obtained an entropy-stable numerical
flux by starting from an entropy-conservative flux
\(f^{\mathrm{ec}}_{i+\frac12}\) and adding a symmetric dissipation term
\(-\tfrac12 D_{i+\frac12}[v]_{i+\frac12}\), which guarantees a discrete
entropy inequality.  It remains to specify how to construct
\(f^{\mathrm{ec}}_{i+\frac12}\) itself. Now, we derive a
practical entropy-conservative flux by enforcing the discrete entropy
condition
\(
[v]_{i+\frac12}^\top f^{\mathrm{ec}}_{i+\frac12} = [\phi(v)]_{i+\frac12}
\)
at each interface with
\([v]_{i+\frac12} := v_{i+\frac12}^+ - v_{i+\frac12}^-\) and
\([\phi(v)]_{i+\frac12} := \phi(v_{i+\frac12}^+) - \phi(v_{i+\frac12}^-)\).

1. Start from a centered flux and add a correction.  
   Take the arithmetic average of the physical fluxes

$$
f^{\mathrm{c}}_{i+\frac12} :=
 \tfrac12\Big(f(u_{i+\frac12}^+)+f(u_{i+\frac12}^-)\Big),
$$

   and look for \(f^{\mathrm{ec}}_{i+\frac12}\) of the form

$$
f^{\mathrm{ec}}_{i+\frac12} =
  f^{\mathrm{c}}_{i+\frac12} +
    \alpha_{i+\frac12}[v]_{i+\frac12},
$$

   where \(\alpha_{i+\frac12}\) is a scalar to be determined.  
   Only the component of the correction parallel to \([v]_{i+\frac12}\)
   can change the scalar product
   \([v]_{i+\frac12}^\top f^{\mathrm{ec}}_{i+\frac12}\), so this is the
   minimal modification needed to enforce the entropy condition.

2. Impose the discrete entropy condition.  
   Plug the ansatz into
   \([v]_{i+\frac12}^\top f^{\mathrm{ec}}_{i+\frac12} =
      [\phi(v)]_{i+\frac12}\):

$$
[v]_{i+\frac12}^\top f^{\mathrm{c}}_{i+\frac12} +
 \alpha_{i+\frac12}\|[v]_{i+\frac12}\|_2^2 =
 [\phi(v)]_{i+\frac12}.
$$

   Hence

$$
\alpha_{i+\frac12}= \frac{[\phi(v)]_{i+\frac12} -
         [v]_{i+\frac12}^\top f^{\mathrm{c}}_{i+\frac12}}
        {\|[v]_{i+\frac12}\|_2^2}.
$$

3. Insert \(\alpha_{i+\frac12}\) back into the flux.  
   Using the expression for \(f^{\mathrm{c}}_{i+\frac12}\), we obtain

$$
f^{\mathrm{ec}}_{i+\frac12} =
  \tfrac12\Big(f(u_{i+\frac12}^+)
               +f(u_{i+\frac12}^-)\Big) +
  \frac{
    [\phi(v)]_{i+\frac12} -
     \tfrac12 [v]_{i+\frac12}^\top
      \Big(f(u_{i+\frac12}^+)+f(u_{i+\frac12}^-)\Big)}
    {\|[v]_{i+\frac12}\|_2^2}
   [v]_{i+\frac12},
$$

At this point we have an explicit interface flux
\(f^{\mathrm{ec}}_{i+\frac12}\) that is entropy-conservative by
construction, i.e. it satisfies
\([v]_{i+\frac12}^\top f^{\mathrm{ec}}_{i+\frac12}=[\phi(v)]_{i+\frac12}\).
This is precisely the baseline flux needed in the discrete entropy condition, adding
the symmetric dissipation term then yields an entropy-stable flux.

