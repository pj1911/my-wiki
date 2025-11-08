# Linear Regression — A brief intro

## Setup

Supervised regression predicts a continuous target $y$ from features $\mathbf{x}$ with model $\hat y = f(\mathbf{x})$.

Loss (mean squared error, MSE):
$$
L(\mathbf{w}) \;=\; \frac{1}{n}\sum_{i=1}^n\!\bigl(y_i-\hat y_i\bigr)^2.
$$

## Baseline: Predict-by-Mean

Constant model $\hat y_i = w_0$ has optimum
$$
w_0^* \;=\; \bar y \;=\; \frac{1}{n}\sum_{i=1}^n y_i.
$$

(Biased) mean and variance:
$$
\bar y=\frac{1}{n}\sum_{i=1}^n y_i,
\qquad
\sigma_y^2=\frac{1}{n}\sum_{i=1}^n (y_i-\bar y)^2.
$$

## Simple (Univariate) Linear Regression

Model with one feature $x$:
$$
\hat y_i = w_0 + w_1 x_i,
\qquad
e_i = y_i - \hat y_i.
$$

Optimal parameters (OLS):
$$
w_1^*=\frac{\sum_{i=1}^n (x_i-\bar x)(y_i-\bar y)}{\sum_{i=1}^n (x_i-\bar x)^2}
=\frac{\sigma_{xy}}{\sigma_x^2}
= r_{xy}\,\frac{\sigma_y}{\sigma_x},
\qquad
w_0^*=\bar y - w_1^*\bar x.
$$

Where the data summaries are
$$
\bar x=\frac{1}{n}\sum_{i=1}^n x_i,\quad
\sigma_x^2=\frac{1}{n}\sum_{i=1}^n (x_i-\bar x)^2,\quad
\sigma_{xy}=\frac{1}{n}\sum_{i=1}^n (x_i-\bar x)(y_i-\bar y),\quad
r_{xy}=\frac{\sigma_{xy}}{\sigma_x\sigma_y}.
$$

Optimal residual variance and $R^2$:
$$
\sigma_e^2
= \sigma_y^2 - \frac{\sigma_{xy}^2}{\sigma_x^2},
\qquad
R^2 = 1 - \frac{\sum_{i=1}^n (y_i-\hat y_i)^2}{\sum_{i=1}^n (y_i-\bar y)^2}
= 1 - \frac{\sigma_e^2}{\sigma_y^2}.
$$

## Multiple Linear Regression

Stack data:

- $\mathbf{X}\in\mathbb{R}^{n\times d}$ (rows are $\mathbf{x}_i^\top$)  
- $\mathbf{y}\in\mathbb{R}^{n}$  
- $\mathbf{A}=\bigl[\mathbf{1}\ \ \mathbf{X}\bigr]\in\mathbb{R}^{n\times(d+1)}$ (first column ones for intercept)

Model and OLS solution (full-rank $\mathbf{A}$):
$$
\hat{\mathbf{y}}=\mathbf{A}\mathbf{w},
\qquad
\mathbf{w}^*=(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{y}.
$$

## Linear Basis Function Regression (LBF)

Replace raw features by fixed basis $\phi_j(\mathbf{x})$:
$$
\hat y_i = \sum_{j=0}^{p} w_j\,\phi_j(\mathbf{x}_i)
= \mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x}_i).
$$

Design matrix:
$$
\boldsymbol{\Phi}=\begin{bmatrix}
\phi_0(\mathbf{x}_1)&\cdots&\phi_p(\mathbf{x}_1)\\
\vdots&\ddots&\vdots\\
\phi_0(\mathbf{x}_n)&\cdots&\phi_p(\mathbf{x}_n)
\end{bmatrix},
\qquad
\mathbf{w}^*=(\boldsymbol{\Phi}^\top\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^\top\mathbf{y}.
$$

## Gradient View (for OLS)

With $L(\mathbf{w})=\tfrac12\|\mathbf{y}-\mathbf{B}\mathbf{w}\|_2^2$ where $\mathbf{B}\in\{\mathbf{A},\boldsymbol{\Phi}\}$:
$$
\nabla L(\mathbf{w})=-\mathbf{B}^\top(\mathbf{y}-\mathbf{B}\mathbf{w})=0
\;\Rightarrow\;
\mathbf{B}^\top\mathbf{B}\,\mathbf{w}=\mathbf{B}^\top\mathbf{y}.
$$

## Interpretation

- Correlation $r_{xy}=\dfrac{\sigma_{xy}}{\sigma_x\sigma_y}\in[-1,1]$ is a data property (no fitting).  
- Coefficient $w_j$:
  - Simple regression: “+1 unit in $x$ $\Rightarrow$ $w_j$ units in $y$ on average.”
  - Multiple regression: effect **holding other included features fixed**.
- Coefficient of determination:
$$
R^2=1-\frac{\sum_{i=1}^n (y_i-\hat y_i)^2}{\sum_{i=1}^n (y_i-\bar y)^2}.
$$

## OLS “Recipe”

1. Collect $(\mathbf{x}_i,y_i)_{i=1}^n$.  
2. Choose model: $\hat y_i=\langle\boldsymbol{\phi}(\mathbf{x}_i),\mathbf{w}\rangle$ (raw linear or with bases).  
3. Choose loss: $L=\frac{1}{n}\sum_i (y_i-\hat y_i)^2$ (or $\tfrac12\|\cdot\|_2^2$ form).  
4. Solve OLS for $\mathbf{w}^*$ via normal equations or a numeric solver.  
5. Predict on new $\mathbf{x}$ using $\hat y=\langle\boldsymbol{\phi}(\mathbf{x}),\mathbf{w}^*\rangle$.  
6. Evaluate (MSE, $R^2$) on held-out data.
