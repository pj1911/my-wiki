## Introduction

A neural operator learns a map between functions, not just between arrays. Formally, if the input is a function

$$
a : D \to \mathbb{R}^{c_{\mathrm{in}}},
$$

and the output is another function

$$
u : D \to \mathbb{R}^{c_{\mathrm{out}}},
$$

then the goal is to learn an operator

$$
\mathcal{G} : a \mapsto u.
$$

This matters for PDEs because the same physical field may be sampled on different grids. A good neural operator should still make sense when the sampling changes.

Fourier Neural Operators (FNOs) are excellent neural operators because they work in function space and can be evaluated at different resolutions. But their main nontrivial branch is global: it mixes information through Fourier modes over the whole domain.
Meanwhile, ordinary CNNs have strong locality bias, which is very useful for PDEs, but a standard discrete convolution is tied to the training grid and does not naturally define the same operator when the grid changes.

The paper's central contribution is to keep the resolution-agnostic operator-learning viewpoint of FNO while adding two genuinely local branches:

1. a differential branch, which behaves like a learned finite-difference stencil in the continuous limit,
2. a local integral branch, which keeps a fixed physical receptive field as the mesh is refined.

### Working Mechanism for FNO

Suppose the hidden feature field at layer \(\ell\) is

$$
v_\ell : D \to \mathbb{R}^m.
$$

An FNO block combines a global spectral convolution with a pointwise linear map, and can be written as

$$
v_{\ell+1}(x) = \sigma\!\left( \underbrace{W_\ell v_\ell(x)}_{\text{pointwise branch}} + \underbrace{\mathcal{F}^{-1}\!\big(R_\ell(k)\,\mathcal{F}v_\ell(k)\big)(x)}_{\text{Fourier branch}} \right).
$$

Here, \(\mathcal{F}\) denotes the Fourier transform, or the Fourier series transform on a periodic domain, \(R_\ell(k)\) is a trainable matrix for each retained Fourier mode \(k\), \(W_\ell\) is a pointwise linear map acting independently at each spatial location, and \(\sigma\) is a nonlinearity such as GELU. Starting from \(v_\ell\), the FNO block transforms the feature field into Fourier space, applies the learned mode-wise matrices \(R_\ell(k)\) to the retained coefficients, maps the result back to physical space via \(\mathcal{F}^{-1}\), combines it with the pointwise term \(W_\ell v_\ell\), and then applies \(\sigma\). Because the Fourier branch is global, the resulting representation at any location \(x\) depends on information from across the entire domain, which makes FNOs particularly effective for modeling long-range interactions and learning operators that transfer across resolutions.

But FNO can still miss important details, since a global spectral representation is often not the most natural way to capture strongly local effects such as derivatives and stencil-based interactions, sharp local fronts, localized kernels, or fine-scale neighborhood structure on irregular meshes. Standard discrete convolutions, on the other hand, are naturally local and therefore effective at capturing neighborhood-level interactions and fine-scale patterns. Yet they cannot be added with the FNO architecture directly without sacrificing key aspects of its operator-learning formulation. The next section makes this precise by showing why an ordinary discrete convolution is not a sufficient remedy.

## Why an ordinary discrete convolution is not enough

This is the most important mathematical observation in the paper. Consider a one-dimensional uniform grid, with uniform distance \(h\) between grid points. In other words, if the grid points are denoted by \(x_j\), then

$$
x_{j+1}-x_j = h
$$

for every \(j\). If we apply a stride-1 convolution with an odd number \(S\) of kernel entries, so the kernel is centered at the current grid point. To describe the locations of the kernel entries relative to the current point \(y\), define

$$
z_i = h\left(i-1-\frac{S-1}{2}\right), \qquad i=1,\dots,S.
$$

For example, if the kernel width \(S=5\), then

$$
z_i = h\left(i-1-\frac{5-1}{2}\right)=h(i-3), \qquad i=1,\dots,5.
$$

So the five offsets are

$$
z_1=-2h,\qquad z_2=-h,\qquad z_3=0,\qquad z_4=h,\qquad z_5=2h.
$$

This means that if the kernel is centered at a point \(y\), then it samples the input at the five locations

$$
y-2h,\qquad y-h,\qquad y,\qquad y+h,\qquad y+2h.
$$

Then the discrete convolution at a grid point \(y\) is

$$
\mathrm{Conv}_K[v]\,(y) = \sum_{i=1}^{5} K_i\, v(y+z_i).
$$

which can be written explicitly as

$$
\mathrm{Conv}_K[v]\,(y) =
K_1v(y-2h)+K_2v(y-h)+K_3v(y)+K_4v(y+h)+K_5v(y+2h).
$$

Here, \(v\) denotes the input function (or feature field), and \(v(y+z_i)\) means the value of \(v\) evaluated at the spatial location \(y+z_i\).

Major issue.

As the mesh is refined, the grid spacing \(h\) decreases. For a fixed kernel width \(S\), the offsets

$$
z_i = h\left(i-1-\frac{S-1}{2}\right)
$$

all shrink to zero as \(h\to 0\). Thus, although the number of grid points in the domain increases, the physical neighborhood sampled by the kernel around a point \(y\) becomes smaller and smaller, so the stencil effectively collapses onto the single point \(y\).
So for a continuous function \(v\),

$$
v(y+z_i) \to v(y).
$$

Therefore,

$$
\mathrm{Conv}_K[v]\,(y)
= \sum_{i=1}^S K_i\, v(y+z_i)
\longrightarrow
\left(\sum_{i=1}^S K_i\right) v(y).
$$

This is only a pointwise linear map. The physical receptive field has collapsed to a point. For example, a \(3\times 3\) kernel is local in pixels, but not necessarily local in physical space. On a finer grid, those same 3 pixels cover a much smaller physical neighborhood. So a standard CNN kernel does not represent one fixed local physical operator across resolutions.

This is why an ordinary discrete convolution is not consistent as a neural operator in function space.

**Suggestions.** To obtain a meaningful local operator in the continuum limit, the construction should act directly in physical space rather than in pixel or grid space, while remaining well behaved under mesh refinement. Motivated by this, the paper proposes two related directions: a local integral-operator viewpoint, which describes nearby interactions through spatially localized kernels, and a differential-operator viewpoint, which connects these local interactions to derivatives in the continuum limit. These ideas are developed in the next section.

## Local integral kernels: keeping the receptive field physically fixed

To incorporate locality into operator learning, we need a notion of interaction that is tied to physical space rather than to a particular discretization. This motivates a continuum formulation in which locality is described directly through kernels acting over neighborhoods in the underlying domain. That viewpoint leads naturally to local integral operators.

### A continuous local averaging operator

We now want to describe a local operator: an operation that updates the value at a point \(y\) by looking only at the input function in a neighborhood around \(y\). A very general way to write such an operator is

$$
(\mathcal{G}_k v)(y) = \int_{U(y)} k(x,y)\,v(x)\,dx,
$$

where:

- \(v(x)\) is the input function,
- \(U(y)\) is a local neighborhood around the point \(y\),
- \(k(x,y)\) is a kernel that tells us how much the value at \(x\) contributes to the output at \(y\).

So, to compute the output at \(y\), we look at all nearby points \(x \in U(y)\), weight each value \(v(x)\) by \(k(x,y)\), and integrate. Since in practice we only know the function on a discrete mesh, this integral is approximated by a sum:

$$
(\mathcal{G}_k v)(y) \approx \sum_{x_j \in D_h \cap U(y)} k(x_j,y)\,v(x_j)\,q_j,
$$

where \(x_j\) are the mesh points and \(q_j\) are quadrature weights that account for the local cell size or area around each point.

### Translation invariance

We now specialize the general local integral operator to the case where the same local rule should be applied at every point in the domain. On a flat domain, this means that the interaction between two points should depend only on their relative displacement, not on their absolute locations.

For this reason, we choose the kernel in the form

$$
k(x,y)=\kappa(x-y).
$$

Here, \(\kappa(x-y)\) depends only on the relative displacement between \(x\) and \(y\), rather than on their absolute positions. This means that the same local weighting rule is used everywhere in the domain, which is exactly the defining idea of convolution: the interaction depends only on offset, not on location. With this choice, the local operator becomes

$$
(\mathcal{I} v)(y)=\int \kappa(x-y)\,v(x)\,dx.
$$

So, to compute the output at \(y\), we gather information from nearby points \(x\), weight each value \(v(x)\) according to its offset from current location \(y\), and integrate. On a mesh, this is approximated by

$$
(\mathcal{I} v)(y)\approx \sum_j \kappa(x_j-y)\,v(x_j)\,q_j,
$$

where \(x_j\) are the mesh points and \(q_j\) are quadrature weights. The key point is that \(\kappa\) is defined as a continuous function of physical offset:

$$
\mathrm{supp}(\kappa)\subset B_r(0)
$$

means that \(\kappa\) is zero outside the ball of radius \(r\). Therefore, only points within distance \(r\) of the center are used, so the operator keeps the same physical neighborhood even as the grid is refined. This is exactly what distinguishes it from an ordinary CNN kernel.

Therefore, as the grid becomes finer, this construction converges to a genuine local integral operator rather than collapsing to a pointwise map.

### How the kernel is parameterized

The paper uses a basis expansion

$$
\kappa(\delta) = \sum_{\ell=1}^L \theta^{(\ell)} \kappa^{(\ell)}(\delta),
$$

where \(\kappa^{(\ell)}\) are fixed basis functions (for example piecewise-linear hat functions), and the trainable parameters are the coefficients \(\theta^{(\ell)}\). Then the layer becomes

$$
(\mathcal{I} v)(y_i) =
\sum_{\ell=1}^L \theta^{(\ell)}
\sum_{j=1}^m \kappa^{(\ell)}(x_j-y_i)v(x_j) q_j,
\qquad
K^{(\ell)}_{ij} = \kappa^{(\ell)}(x_j-y_i).
$$

Because the kernel \(\kappa\) is supported only in a small neighborhood, \(\kappa(x_i,x_j)\) is nonzero only when \(x_i\) and \(x_j\) are close. As a result, most entries of the matrix \(K_{ij}\) are zero, so the matrix is sparse.

## Differential kernels: from convolution to derivatives

Local integral kernels describe interactions over a fixed physical neighborhood, but many PDEs involve a more specific class of local operators: derivatives. Gradients, divergences, and Laplacians measure how a field changes in space, so a useful local neural-operator layer should be able to recover derivative-like behavior in the continuum limit. The goal of this section is to make that connection precise.

A basic example is the centered finite difference

$$
\frac{v(y+h)-v(y-h)}{2h} \approx v'(y),
$$

which may be viewed as a local stencil with weights \([-1,0,1]\) scaled by \(1/(2h)\), that cancel constant functions and isolate first-order variation. The main idea in the paper is to extend this principle from hand-designed finite-difference formulas to learned local operators: instead of fixing the stencil coefficients in advance, one learns them from data while enforcing the structural constraints required for a meaningful differential limit as the mesh is refined.

### The key construction

To obtain a local operator that behaves like a derivative rather than an averaging rule, we start from a centered stencil and impose a zero-sum condition on its weights. Let

$$
\sum_{i=1}^S a_i = 0,
$$

and consider the operator

$$
(\mathcal{D}_h v)(y)=\frac{1}{h}\sum_{i=1}^S a_i\,v(y+z_i),
$$

where, as in the previous section, the offsets \(z_i\) describe the local stencil around \(y\). The factor \(1/h\) gives the correct first-order scaling, while the condition

$$
\sum_{i=1}^S a_i = 0
$$

removes the zeroth-order contribution. In particular, if \(v\) is constant, then \((\mathcal{D}_h v)(y)=0\), so the operator does not act like local averaging.

This cancellation is the essential mechanism that makes the stencil sensitive to variation in \(v\). As the mesh is refined, the operator therefore captures local changes of the field and can converge to a spatial derivative in the continuum limit.

### Why the derivative appears: scalar 1D derivation

To see why \(\mathcal{D}_h\) converges to a derivative, assume that \(v\) is smooth and expand \(v(y+z_i)\) around \(y\). By Taylor's theorem,

$$
v(y+z_i) = v(y)+z_i v'(y)+\frac12 z_i^2 v''(\xi_i),
$$

where \(\xi_i\) lies between \(y\) and \(y+z_i\).

Substituting this into

$$
(\mathcal{D}_h v)(y)=\frac{1}{h}\sum_{i=1}^S a_i\,v(y+z_i),
$$

we obtain

$$
(\mathcal{D}_h v)(y) = \frac{1}{h}\sum_{i=1}^S a_i \left( v(y)+z_i v'(y)+\frac12 z_i^2 v''(\xi_i) \right)
$$

and therefore

$$
(\mathcal{D}_h v)(y) = \frac{v(y)}{h}\sum_{i=1}^S a_i + \frac{v'(y)}{h}\sum_{i=1}^S a_i z_i + \frac{1}{2h}\sum_{i=1}^S a_i z_i^2 v''(\xi_i).
$$

We now examine these three terms.

Zeroth-order term.

By construction,

$$
\sum_{i=1}^S a_i = 0,
$$

so the first term vanishes:

$$
\frac{v(y)}{h}\sum_{i=1}^S a_i = 0.
$$

First-order term.

Recall that the stencil offsets satisfy

$$
z_i = h\left(i-1-\frac{S-1}{2}\right).
$$

Define the dimensionless offsets

$$
\alpha_i := i-1-\frac{S-1}{2},
\qquad\text{so that}\qquad
z_i = h\alpha_i.
$$

Then

$$
\frac{v'(y)}{h}\sum_{i=1}^S a_i z_i = \frac{v'(y)}{h}\sum_{i=1}^S a_i (h\alpha_i) = v'(y)\sum_{i=1}^S a_i \alpha_i.
$$

Thus the first-order contribution is a constant multiple of \(v'(y)\), independent of \(h\).

Remainder term.

Using again \(z_i=h\alpha_i\), we have

$$
\frac{1}{2h}\sum_{i=1}^S a_i z_i^2 v''(\xi_i) = \frac{1}{2h}\sum_{i=1}^S a_i h^2\alpha_i^2 v''(\xi_i) = \frac{h}{2}\sum_{i=1}^S a_i \alpha_i^2 v''(\xi_i).
$$

If \(v''\) is bounded, this term is \(O(h)\), and therefore vanishes as \(h\to 0\).

Combining the three parts gives

$$
(\mathcal{D}_h v)(y) = \left(\sum_{i=1}^S a_i \alpha_i\right)v'(y) + O(h).
$$

Hence, as \(h\to 0\),

$$
(\mathcal{D}_h v)(y)\to c\,v'(y),
\qquad
c:=\sum_{i=1}^S a_i \alpha_i.
$$

So the learned centered stencil converges to a first-order differential operator in the continuum limit.

### Extension to multiple dimensions and channels

The same idea extends directly to vector-valued fields on \(\mathbb{R}^d\). Let

$$
v=(v_1,\dots,v_m):D\to\mathbb{R}^m.
$$

Applying the Taylor expansion componentwise shows that a centered local stencil with the appropriate first-moment scaling converges to a linear combination of first derivatives of the components of \(v\). In the limit \(h\to 0\), one obtains an operator of the form

$$
\mathcal{D}_h v(y)\;\longrightarrow\;\sum_{j=1}^m \nabla v_j(y)\cdot b_j,
$$

where each \(b_j\in\mathbb{R}^d\) is determined by the learned stencil coefficients and encodes a direction in physical space.

Thus, in several spatial dimensions and across multiple channels, the learned local kernel still converges to a first-order differential operator. The scalar one-dimensional derivation above is therefore the simplest instance of a more general phenomenon: learned centered stencils recover directional derivative structure in the continuum limit.

### The moment-matching viewpoint

The appendix gives a more general expansion for local kernels. For a sufficiently smooth function \(v\),

$$
\sum_{x\in B_h(y)\cap D_h} k(x,y)\,v(x) = v(y)\sum_x k(x,y) + \nabla v(y)\cdot \sum_x k(x,y)(x-y) + O(h).
$$

This formula separates the local action of the kernel into a zeroth-order part and a first-order part. The zeroth-order term is controlled by the total mass of the kernel,

$$
\sum_x k(x,y),
$$

while the first-order term is controlled by its first moment,

$$
\sum_x k(x,y)(x-y).
$$

Therefore, if we want the continuum limit to have the form

$$
c\,v(y)+\nabla v(y)\cdot b,
$$

then the kernel must satisfy the moment conditions

$$
\sum_x k(x,y)=c,
\qquad
\sum_x k(x,y)(x-y)=b.
$$

For a purely differential operator, one sets

$$
c=0,
$$

so that the zeroth-order contribution vanishes and only the first-order term remains. This is the abstract reason the centered differential kernel works: it cancels local averaging while retaining the leading derivative term.

### Practical interpretation

This branch is especially useful when the target operator is itself differential, or when the relevant interactions are strongly local in physical space. In such settings, the differential kernel provides an inductive bias that is better aligned with the underlying PDE structure than a purely global spectral representation.

## Model Architecture

The paper extends each FNO block by adding local operator branches to the usual global spectral update. A convenient summary of one layer is

$$
v_{\ell+1}(x) = \sigma\!\left( \underbrace{\mathcal{W}_\ell v_\ell(x)}_{\text{pointwise branch}} + \underbrace{\mathcal{K}^{\mathrm{Fourier}}_\ell v_\ell(x)}_{\text{global Fourier branch}} + \underbrace{\mathcal{I}_\ell v_\ell(x)}_{\text{local integral branch}} + \underbrace{\mathcal{D}_\ell v_\ell(x)}_{\text{local differential branch}} \right).
$$

Thus, each block combines four complementary mechanisms: pointwise channel mixing, global information exchange through Fourier modes, local averaging over a fixed physical neighborhood, and local derivative-like interactions. Depending on the experiment, some of these branches may be omitted by setting the corresponding term to zero.

### What each branch contributes

Each term in the block plays a distinct role:

| Branch | Role |
| --- | --- |
| Pointwise \(\mathcal{W}_\ell\) | Mixes channels independently at each spatial location, without coupling nearby points. |
| Fourier \(\mathcal{K}^{\mathrm{Fourier}}_\ell\) | Provides global communication across the entire domain through the spectral representation. |
| Local integral \(\mathcal{I}_\ell\) | Aggregates information from a fixed neighborhood in physical space. |
| Differential \(\mathcal{D}_\ell\) | Captures local variation and derivative-like structure. |

## How information flows through the architecture

Suppose the input is a PDE state, such as vorticity, pressure, geopotential height, or velocity, represented as a function

$$
a:D\to\mathbb{R}^{c_{\mathrm{in}}}.
$$

As in FNO, the network first lifts this input pointwise into a latent feature field

$$
v_0(x)=\mathcal{P}(a(x)),
\qquad
v_0(x)\in\mathbb{R}^m,
$$

where \(\mathcal{P}\) is typically a pointwise linear map. The purpose of this lifting step is to convert the raw physical variables into a richer representation on which the operator blocks act.

Each subsequent layer updates the latent field by combining the four branches introduced earlier:

$$
v_{\ell+1}(x) = \sigma\!\left( \mathcal{W}_\ell v_\ell(x) + \mathcal{K}_\ell^{\mathrm{Fourier}} v_\ell(x) + \mathcal{I}_\ell v_\ell(x) + \mathcal{D}_\ell v_\ell(x) \right).
$$

The pointwise branch \(\mathcal{W}_\ell\) mixes channels at the same spatial location. The Fourier branch \(\mathcal{K}_\ell^{\mathrm{Fourier}}\) provides global communication by transforming \(v_\ell\) to Fourier space, applying learned mode-wise matrices to the retained coefficients, and transforming back to physical space. The local integral branch \(\mathcal{I}_\ell\) aggregates information from a neighborhood whose size is fixed in physical space, while the differential branch \(\mathcal{D}_\ell\) applies a centered local stencil that captures derivative-like structure.

After these branch outputs are computed, they are added together and passed through the nonlinearity \(\sigma\). Stacking many such layers allows the model to build a nonlinear operator from repeated combinations of global spectral interactions, local neighborhood aggregation, local derivative information, and pointwise channel mixing.

After \(L\) layers, the final latent field is projected back to the target variables:

$$
u(x)=\mathcal{Q}\bigl(v_L(x)\bigr),
$$

where \(\mathcal{Q}\) is again typically a pointwise map. In this way, the network learns the operator

$$
a\mapsto u
$$

by repeatedly combining global and local mechanisms at every layer.

### Interpretation at a single spatial point

It is useful to interpret one update at a fixed point \(y\). At that point, the Fourier branch brings in information from the entire domain, the local integral branch aggregates values from a neighborhood around \(y\), the differential branch measures how the feature field changes near \(y\), and the pointwise branch simply remixes the channels already present at \(y\). Thus, each update combines four different kinds of information:

$$
\text{global context} \;+\; \text{local neighborhood information} \;+\; \text{local derivative structure} \;+\; \text{pointwise channel mixing}
$$

This is the main architectural idea of the paper: retain the global operator-learning strength of FNO while enriching each layer with local mechanisms that remain meaningful in the continuum limit.

## Why this helps

The benefit of this architecture is that many PDEs involve both global and local structure, so a hybrid model is often a better match than a purely global one. The Fourier branch captures long-range interactions across the domain, while the local branches preserve nearby structure that is often crucial in PDEs. In particular, the differential branch provides a natural inductive bias for operators involving derivatives, since learned centered stencils are closely aligned with how classical numerical methods approximate gradients, divergences, and Laplacians. The local integral branch, meanwhile, preserves a fixed physical receptive field as the mesh changes, so it retains locality without losing the resolution-transfer property of neural operators. Because these local kernels depend on relative position, they also inherit the parameter sharing and translation-equivariant bias associated with convolution. Finally, the same framework extends beyond regular Cartesian grids to spheres and irregular meshes, which is essential for many scientific applications where the underlying geometry is non-Euclidean.

## Conclusion

Main idea.

The central insight of the paper is that a local operation should remain local in physical space when the discretization changes. A standard discrete convolution does not satisfy this requirement. The paper resolves this in two principled ways. First, by centering and scaling a local stencil, it constructs a learned differential operator whose limit is derivative-like. Second, by defining the kernel as a continuous function of physical offset, it obtains a learned local integral operator whose receptive field remains fixed in physical space across resolutions.

Architectural contribution.

These local constructions are then combined with the usual FNO/Spherical FNO global branch. The resulting architecture brings together four complementary ingredients: global communication through Fourier layers, local neighborhood aggregation through integral kernels, local derivative modeling through differential kernels, and pointwise channel mixing. This makes the model a better match for many PDEs, which often involve both long-range structure and strongly local interactions. In particular, the differential branch provides an inductive bias well aligned with operators containing gradients, divergences, or Laplacians, while the local integral branch preserves locality without sacrificing the cross-resolution viewpoint of neural operators.

Important caveats.

The paper also makes clear that these advantages do not remove all numerical difficulties. Although the construction is discretization-agnostic in principle, performance can still depend strongly on the training resolution if the data does not adequately resolve the relevant local physics. Differential layers can accumulate discretization error when stacked repeatedly, especially at insufficient resolution, in the Darcy super-resolution experiments in the paper, the strongest variant used a differential layer only in the first block. Boundary effects also do not disappear automatically, particularly in non-periodic settings where FFT-based processing is less natural. Finally, the cost of the local integral branch depends on the size of its physical support: as that support covers more neighbors on finer grids, computation increases, even if it remains more efficient than generic graph-based operator layers.

Final takeaway.

Overall, the contribution of the paper is to enrich the neural-operator framework with local mechanisms that remain meaningful in the continuum limit. The resulting model is not simply FNO with extra local layers. It is a mathematically motivated way to make locality compatible with operator learning. That is why it can improve on both purely global spectral models and standard grid-tied convolutions: it matches the structure of many PDE operators more faithfully.
