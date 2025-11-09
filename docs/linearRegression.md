# Linear Regression — A Brief, Friendly Intro

## What problem are we solving?

Suppose we have examples $(\mathbf{x}_i, y_i)$ for $i=1,\dots,n$. We want to predict an output $y$ from an input feature vector $\mathbf{x}$. Let $f(\cdot)$ be a model that makes predictions $\hat y$ from $x$, so $\hat y = f(x)$. We judge our model by how close $\hat y$ is to $y$.

**How wrong we are (MSE):**
$$
L=\frac{1}{n}\sum_{i=1}^n \bigl(y_i-\hat y_i\bigr)^2,
\qquad
\hat y_i=f(x_i).
$$

## A tiny grounding example

Suppose $x$ = study hours and $y$ = test score for three students:

| $i$ | $x_i$ (hours) | $y_i$ (score) |
|---:|--------------:|--------------:|
| 1 | 1 | 2 |
| 2 | 2 | 3 |
| 3 | 3 | 5 |

We’ll use this toy set to make the formulas concrete.

## Baseline: always predict the average

If we must make a constant prediction for everyone ($\hat y_i = f(x_i)=w_0$), the best constant ($w_0^*$) is the average of the training targets. Minimize the MSE:
$$
L=\frac{1}{n}\sum_{i=1}^n \bigl(y_i-w_0\bigr)^2,
\qquad
\frac{\partial L}{\partial w_0}=0
\ \Rightarrow\
w_0^*=\bar y=\frac{1}{n}\sum_{i=1}^n y_i.
$$
This do-nothing baseline matters because any smart model should beat it.

## Simple (one-feature) linear regression

Now let the prediction depend on $x$:
$$
\hat y_i = w_0 + w_1 x_i.
$$
Define residuals $e_i=y_i-\hat y_i$ so that $y_i=\hat y_i+e_i$ (the error term captures what the model misses).

**Optimal parameters $w_0^*$ and $w_1^*$:**
$$
L=\frac{1}{n}\sum_{i=1}^n \bigl(y_i-w_0-w_1x_i\bigr)^2.
$$
Minimizing MSE gives the ordinary least squares (OLS) solution
$$
w_1^*=\frac{\sum (x_i-\bar x)(y_i-\bar y)}{\sum (x_i-\bar x)^2}
=\frac{\sigma_{xy}}{\sigma_x^2}
= r_{xy}\frac{\sigma_y}{\sigma_x},
\qquad
w_0^*=\bar y-w_1^*\bar x,
$$
where
$$
\bar x=\frac{1}{n}\sum x_i,\quad
\sigma_x^2=\frac{1}{n}\sum (x_i-\bar x)^2,\quad
\sigma_{xy}=\frac{1}{n}\sum (x_i-\bar x)(y_i-\bar y),\quad
r_{xy}=\frac{\sigma_{xy}}{\sigma_x\sigma_y}.
$$

### Metrics for goodness of fit

Residuals are $e_i=y_i-\hat y_i$ with $\hat y_i=w_0+w_1x_i$.

- **Residual variance (biased):**
  $$
  \sigma_e^2=\frac{1}{n}\sum_{i=1}^n e_i^2.
  $$
  (Biased because it divides by $n$.)

- **Unbiased residual variance (simple regression with intercept):**
  $$
  s_e^2=\frac{1}{n-2}\sum_{i=1}^n e_i^2.
  $$

Start from the least-squares objective
$$
L(w_0,w_1)=\sum_{i=1}^n\bigl(y_i-w_0-w_1x_i\bigr)^2.
$$
Set partial derivatives to zero:
$$
\frac{\partial L}{\partial w_0}=-2\sum (y_i-w_0-w_1x_i)=0,\qquad
\frac{\partial L}{\partial w_1}=-2\sum x_i(y_i-w_0-w_1x_i)=0.
$$
These yield the normal equations
$$
\sum e_i=0
\ \Rightarrow\
w_0=\bar y - w_1\bar x,
\qquad
\sum x_ie_i=0
\ \Rightarrow\
w_1=\frac{\sum (x_i-\bar x)(y_i-\bar y)}{\sum (x_i-\bar x)^2}.
$$

Define
$$
S_{xx}=\sum (x_i-\bar x)^2,\quad
S_{yy}=\sum (y_i-\bar y)^2,\quad
S_{xy}=\sum (x_i-\bar x)(y_i-\bar y),
$$
so $w_1=S_{xy}/S_{xx}$ and $w_0=\bar y-w_1\bar x$. Substituting $w_0$ gives
$$
e_i=(y_i-\bar y)-w_1(x_i-\bar x).
$$
Then the sum of squared errors (SSE) is
$$
\sum e_i^2
= S_{yy} - 2w_1 S_{xy} + w_1^2 S_{xx}
= S_{yy} - \frac{S_{xy}^2}{S_{xx}}.
$$
Divide by $n$ to match the “population” convention:
$$
\sigma_e^2
= \frac{1}{n}\sum e_i^2
= \sigma_y^2 - \frac{\sigma_{xy}^2}{\sigma_x^2}.
$$

Finally, with $R^2=1-\dfrac{\text{SSE}}{\text{SST}}=1-\dfrac{\sum e_i^2}{\sum (y_i-\bar y)^2}$,
$$
R^2
= 1-\frac{S_{yy}-\frac{S_{xy}^2}{S_{xx}}}{S_{yy}}
= \frac{S_{xy}^2}{S_{xx}\,S_{yy}}
= \frac{\sigma_{xy}^2}{\sigma_x^2\,\sigma_y^2}
= r_{xy}^2.
$$

**How do we know if the fit is good?**
- **Residual variance** $\sigma_e^2$ (or $\mathrm{RMSE}=\sqrt{\sigma_e^2}$) is **small relative to the scale of $y$**. Rule of thumb: $\mathrm{RMSE}\ll \sigma_y$ is good.
- **$R^2$ close to 1** means the model explains most variance.
- **Beat the baseline:** ensure $\sigma_e^2<\sigma_y^2$ (equivalently $R^2>0$). If not, the mean-only model is better.
- **Out-of-sample:** check test/validation $R^2$ or RMSE. A good fit **generalizes** (train and test are similar).
- **Residual diagnostics:** residuals should look like noise (no trend vs.\ $\hat y$ or $x$, roughly constant spread, few large outliers).
- **Adjusted $R^2$ (for multiple $x$):** prefer higher adjusted $R^2$; it penalizes unnecessary features.

## Multiple linear regression (many features)

With $d$ features per example:
$$
\mathbf{X}\in\mathbb{R}^{n\times d},\quad
\mathbf{y}\in\mathbb{R}^{n},\quad
\mathbf{A}=\bigl[\mathbf{1}\ \ \mathbf{X}\bigr]\in\mathbb{R}^{n\times(d+1)}.
$$
Per sample:
$$
\hat y_i = w_0 + w_1x_{i,1} + \cdots + w_d x_{i,d}.
$$
Vector form:
$$
\hat{\mathbf{y}}=\mathbf{A}\mathbf{w},\qquad
\mathbf{w}^*=(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{y}.
$$
Each coefficient means: **change in $y$ for a one-unit change in that feature, holding the other included features fixed.**

## Linear basis function regression (same idea, richer inputs)

Rather than raw features, feed fixed transformations (polynomials, splines, one-hot bins, etc.):
$$
\hat y_i=\sum_{j=0}^{p} w_j\,\phi_j(\mathbf{x}_i)
= \mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x}_i),\qquad \phi_0(\cdot)=1.
$$
Matrix form:
$$
\boldsymbol{\Phi}=
\begin{bmatrix}
\phi_0(\mathbf{x}_1)&\cdots&\phi_p(\mathbf{x}_1)\\
\vdots&\ddots&\vdots\\
\phi_0(\mathbf{x}_n)&\cdots&\phi_p(\mathbf{x}_n)
\end{bmatrix},
\qquad
\mathbf{w}^*=(\boldsymbol{\Phi}^\top\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^\top\mathbf{y}.
$$
Same solver, different design matrix.

## Why the normal equations appear (gradient view)

Write the loss as a quadratic:
$$
L(\mathbf{w})=\tfrac12\|\mathbf{y}-\mathbf{B}\mathbf{w}\|_2^2,\quad
\mathbf{B}\in\{\mathbf{A},\boldsymbol{\Phi}\}.
$$
Set the gradient to zero:
$$
\nabla L(\mathbf{w})=-\mathbf{B}^\top(\mathbf{y}-\mathbf{B}\mathbf{w})=0
\ \Rightarrow\
\mathbf{B}^\top\mathbf{B}\,\mathbf{w}=\mathbf{B}^\top\mathbf{y}.
$$
That linear system **is** the normal equations above.

## Quick OLS recipe (practical path)

1. Gather data $(\mathbf{x}_i,y_i)_{i=1}^n$ **including an intercept column of ones**.
2. Choose features/bases $\boldsymbol{\phi}(\cdot)$ (raw, polynomial, binned, etc.).
3. Fit by OLS: $\mathbf{w}^*=(\mathbf{B}^\top\mathbf{B})^{-1}\mathbf{B}^\top\mathbf{y}$, or use a numeric solver (QR/SVD are more stable).
4. Predict: $\hat y=\langle\boldsymbol{\phi}(\mathbf{x}),\mathbf{w}^*\rangle$.
5. Evaluate on held-out data (MSE, $R^2$).
