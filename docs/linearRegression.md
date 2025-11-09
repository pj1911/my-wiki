# Linear Regression — Intro

## What problem are we solving?

Suppose we have examples \((\mathbf{x}_i, y_i)\) where \(i = 1,...,n\), so we have n total examples and we want to predict an output \(y\) from the input feature \(\mathbf{x}\). Let us say that we also have a model \(f(\mathbf{.})\) that makes predictions \(\hat y\) as some function of the input x, giving \(\hat{y} = f(x)\). We judge our model by how close \(\hat y\) is to \(y\).

**How wrong we are:**  
The most common way to judge our model is to check it's mean squared error (MSE), given by:
\[
L=\frac{1}{n}\sum_{i=1}^n \bigl(y_i-\hat y_i\bigr)^2.
\]
where, \(\hat{y_i} = f(x_i)\)

## A tiny grounding example

Suppose \(x\) = study hours and \(y\) = test score for three students:

| \(i\) | \(x_i\) (hours) | \(y_i\) (score) |
|-----:|:---------------:|:---------------:|
| 1    | 1               | 2               |
| 2    | 2               | 3               |
| 3    | 3               | 5               |

We will use this toy set to make the formulas concrete.

## Baseline: always predict the average

If we must make a constant prediction for everyone \((\hat{y_i} =f(x_i) =w_0)\), the best constant \((w_0^*)\) is the average of the training targets, this can be proved as follows. We had our models MSE as:
\[
L=\frac{1}{n}\sum_{i=1}^n \bigl(y_i-w_0\bigr)^2.
\]
We try to minimize this MSE to find the best prediction \((w_0^*)\)
\[
\frac{\partial L}{\partial w_0}=0.
\]
This gives:
\[
w_0^*=\bar y=\frac{1}{n}\sum_{i=1}^n y_i.
\]
This ``do-nothing'' baseline matters because any smart model should beat it, meaning the predictions should be better than this baseline model.

## Simple (one-feature) linear regression

Now let the prediction depend on \(x\) instead of just being constant, so \(\hat{y_i} =f(x_i) \neq w_0\) anymore, but:
\[
\hat y_i = w_0 + w_1 x_i
\]
We can then define the error for each sample i as: \(e_i = y_i-\hat y_i\). Then we can write \(y_i = \hat y_i + e_i\). This means the error term captures everything from the reality that the model cannot.

**Optimal Parameters \(w_0^*\) and \( w_1^*\):**  
We can write the MSE in this case as:
\[
L=\frac{1}{n}\sum_{i=1}^n \bigl(y_i-w_0-w_1x_i\bigr)^2.
\]
Minimizing MSE gives the ordinary least squares (OLS) solution
\[
w_1^*=\frac{\sum (x_i-\bar x)(y_i-\bar y)}{\sum (x_i-\bar x)^2}
=\frac{\sigma_{xy}}{\sigma_x^2}
= r_{xy}\frac{\sigma_y}{\sigma_x},
\qquad
w_0^*=\bar y-w_1^*\bar x,
\]
where,
\[
\bar x=\frac{1}{n}\sum x_i,\quad
\sigma_x^2=\frac{1}{n}\sum (x_i-\bar x)^2,\quad
\sigma_{xy}=\frac{1}{n}\sum (x_i-\bar x)(y_i-\bar y),\quad
r_{xy}=\frac{\sigma_{xy}}{\sigma_x\sigma_y}.
\]

**Metrics for goodness of fit**  
We know the residuals \(e_i=y_i-\hat y_i\) with \(\hat y_i=w_0+w_1x_i\), and their (biased) variance is given by:
\[
\sigma_e^2=\frac{1}{n}\sum_{i=1}^n e_i^2.
\]
We call it \emph{biased} because it divides by \(n\). The \emph{unbiased} version (simple regression with intercept) is:
\[
s_e^2=\frac{1}{n-2}\sum_{i=1}^n e_i^2.
\]
Start from the least squares objective:
\[
L(w_0,w_1)=\sum_{i=1}^n\bigl(y_i-w_0-w_1x_i\bigr)^2.
\]
Set the partial derivatives to zero:
\[
\frac{\partial S}{\partial w_0}=-2\sum (y_i-w_0-w_1x_i)=0,\qquad
\frac{\partial S}{\partial w_1}=-2\sum x_i(y_i-w_0-w_1x_i)=0.
\]
These two equations give the normal equations
\[
\sum e_i=0
\ \Rightarrow\ 
w_0=\bar y - w_1\bar x,
\qquad
\sum x_ie_i=0
\ \Rightarrow\ 
w_1=\frac{\sum (x_i-\bar x)(y_i-\bar y)}{\sum (x_i-\bar x)^2}.
\]
Define the basic sums
\[
S_{xx}=\sum (x_i-\bar x)^2,\quad
S_{yy}=\sum (y_i-\bar y)^2,\quad
S_{xy}=\sum (x_i-\bar x)(y_i-\bar y),
\]
so \(w_1=S_{xy}/S_{xx}\) and \(w_0=\bar y-w_1\bar x\). Substituting \(w_0\) we get:
\[
e_i
= y_i - w_0 - w_1x_i
= (y_i-\bar y) - w_1(x_i-\bar x).
\]
Therefore the (biased) residual variance times \(n\) (i.e., SSE) is
\[
\sum e_i^2
=\sum\bigl[(y_i-\bar y)-w_1(x_i-\bar x)\bigr]^2
= S_{yy} - 2w_1 S_{xy} + w_1^2 S_{xx}.
\]
Substitute \(w_1=S_{xy}/S_{xx}\) and simplify:
\[
\sum e_i^2
= S_{yy} - 2\frac{S_{xy}^2}{S_{xx}} + \frac{S_{xy}^2}{S_{xx}}
= S_{yy} - \frac{S_{xy}^2}{S_{xx}}.
\]
Divide by \(n\) to match the ``population'' convention used for \(\sigma_e^2\):
\[
\sigma_e^2
= \frac{1}{n}\sum e_i^2
= \frac{S_{yy}}{n} - \frac{S_{xy}^2}{n\,S_{xx}}
= \sigma_y^2 - \frac{\sigma_{xy}^2}{\sigma_x^2}.
\]

Finally, define \(R^2=1-\dfrac{\text{SSE}}{\text{SST}}=1-\dfrac{\sum e_i^2}{\sum (y_i-\bar y)^2}\) to get
\[
R^2
= 1-\frac{S_{yy}-\frac{S_{xy}^2}{S_{xx}}}{S_{yy}}
= \frac{S_{xy}^2}{S_{xx}\,S_{yy}}
= \frac{\sigma_{xy}^2}{\sigma_x^2\,\sigma_y^2}
= r_{xy}^2.
\]

**How do we know if the fit is good?**
- \textbf{Residual variance} \(\boldsymbol{\sigma_e^2}\) (or RMSE \(=\sqrt{\sigma_e^2}\)) should be \emph{small relative to the scale of \(y\)}.  
  Rule of thumb: \(\mathrm{RMSE} \ll \text{SD}(y)=\sigma_y\) is good.
- \textbf{\(\boldsymbol{R^2}\)} close to \(1\) means the model explains most variance.
- \textbf{Beating the baseline}: ensure \(\sigma_e^2 < \sigma_y^2\) (equivalently \(R^2>0\)). If not, the mean-only model is better.
- \textbf{Out-of-sample}: check test/validation \(R^2\) or RMSE. A good fit \emph{generalizes} (train and test metrics are similar).
- \textbf{Residual diagnostics}: residuals should look like noise (no trend vs.\ \(\hat y\) or \(x\), roughly constant spread, few large outliers).
- \textbf{Adjusted \(R^2\) (for multiple \(x\))}: prefer higher adjusted \(R^2\); it penalizes unnecessary features.

## Multiple linear regression (many features)

When we have multiple input features \((d)\) per example, stack them in a matrix:
\[
\mathbf{X}\in\mathbb{R}^{n\times d} \quad
\mathbf{y}\in\mathbb{R}^{n},\quad
\mathbf{A}=\bigl[\mathbf{1}\ \ \mathbf{X}\bigr]\in\mathbb{R}^{n\times(d+1)}.
\]
For each sample we have:
\[
\hat{y_i} = w_0 + w_1x_{i,1} + \cdots + w_dx_{i,d}
\]
Then the model can be written as :
\[
\hat{\mathbf{y}}=\mathbf{A}\mathbf{w},
\]
Then minimizing gives the solution \(w^*\) as:
\[
\mathbf{w}^*=(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{y}.
\]
Each coefficient now means: \emph{change in \(y\) for a one-unit change in that feature, holding the other included features fixed.}

## Linear basis function regression (same idea with richer inputs)

The assumption that the output is a linear function of the input features is very restrictive. Instead, we can consider them to be linear combinations of fixed non-linear functions. Therefore, instead of feeding raw features, we can feed any fixed transformations: polynomials, splines, one-hot bins, etc.
\[
\hat y_i=\sum_{j=0}^{p} w_j\,\phi_j(\mathbf{x}_i)
= \mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x}_i),\qquad \phi_0(\cdot)=1.
\]
In matrix form,
\[
\boldsymbol{\Phi}=
\begin{bmatrix}
\phi_0(\mathbf{x}_1)&\cdots&\phi_p(\mathbf{x}_1)\\
\vdots&\ddots&\vdots\\
\phi_0(\mathbf{x}_n)&\cdots&\phi_p(\mathbf{x}_n)
\end{bmatrix},
\]
Again, minimizing gives:
\[
\mathbf{w}^*=(\boldsymbol{\Phi}^\top\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^\top\mathbf{y}
\]
Same solver, just a different design matrix.

## Why the normal equations appear (gradient view)

Write the loss as a clean quadratic:
\[
L(\mathbf{w})=\tfrac12\|\mathbf{y}-\mathbf{B}\mathbf{w}\|_2^2,\quad
\mathbf{B}\in\{\mathbf{A},\boldsymbol{\Phi}\}.
\]
Setting the gradient to zero:
\[
\nabla L(\mathbf{w})=-\mathbf{B}^\top(\mathbf{y}-\mathbf{B}\mathbf{w})=0
\ \Rightarrow\
\mathbf{B}^\top\mathbf{B}\,\mathbf{w}=\mathbf{B}^\top\mathbf{y}.
\]
That linear system \emph{is} the normal equations above.

## Quick OLS recipe (practical path)

1. Gather data \((\mathbf{x}_i,y_i)_{i=1}^n\) \emph{including an intercept column of ones}.
2. Choose your features/bases \(\boldsymbol{\phi}(\cdot)\) (raw, polynomial, binned, etc.).
3. Fit by OLS: \(\mathbf{w}^*=(\mathbf{B}^\top\mathbf{B})^{-1}\mathbf{B}^\top\mathbf{y}\) or a numeric solver (QR/SVD are more stable).
4. Predict: \(\hat y=\langle\boldsymbol{\phi}(\mathbf{x}),\mathbf{w}^*\rangle\).
5. Evaluate on held-out data (MSE, \(R^2\)).
