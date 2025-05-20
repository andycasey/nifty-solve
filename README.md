<div align="Center">

# nifty-solve

Fit very flexible linear models using Fourier bases without ever constructing the design matrix.

[![Test Status](https://github.com/andycasey/nifty-solve/actions/workflows/ci.yml/badge.svg)](https://github.com/andycasey/nifty-solve/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/andycasey/nifty-solve/badge.svg?branch=main&service=github)](https://coveralls.io/github/andycasey/nifty-solve?branch=main)

</div>

# Install

From source:

```
git clone git@github.com:andycasey/nifty-solve.git
```


# Examples



## 1D real-valued signal with unknown uncertainties

```python
import numpy as np
import matplotlib.pyplot as plt
from nifty_solve.operators import Finufft1DRealOperator

np.random.seed(1)

N = 128 # number of data points
K = 5 # number of Fourier modes

# Generate data
t = np.random.uniform(size=N)
y = np.random.uniform(size=3) @ np.array([t**2, t, np.ones(N)]) + np.random.normal(size=N) * 0.02

# Create a linear operator and solve the system
A = Finufft1DRealOperator(t, n_modes=K)
θ = A / y

# Make a plot
ti = np.linspace(0, 1, 1000)
Ai = Finufft1DRealOperator(ti, n_modes=K)

# Make a plot
fig, ax = plt.subplots()
ax.scatter(t, y, c="k")
ax.plot(ti, Ai @ θ)
ax.set_xlabel("t")
ax.set_ylabel("y")
```
![figure1](https://github.com/user-attachments/assets/207b1aa6-5326-40c6-9101-8340d6092370)

> [!TIP]
> The `A / y` is just syntactic sugar for finding the least-squares solution:
>
> ```python
> from scipy.sparse.linalg import lsqr
>
> θ_1 = A / y
> θ_2, *extras = lsqr(A, y)
> assert np.allclose(θ_1, θ_2)
> ```

## 1D real-valued signal with uncertainties

```python
import numpy as np
import matplotlib.pyplot as plt
from nifty_solve.operators import Finufft1DRealOperator
from pylops import Diagonal

np.random.seed(1)

N = 128 # number of data points
K = 5 # number of Fourier modes

# Generate data
t = np.random.uniform(size=N)
y_true = np.random.uniform(size=3) @ np.array([t**2, t, np.ones(N)])
y_err = 0.05 + np.abs(np.random.normal(size=N) * 0.02)
Y = y_true + y_err * np.random.normal(size=N)
C_inv = Diagonal(y_err**-2)

# Create a linear operator and solve the system
A = Finufft1DRealOperator(t, n_modes=K)
θ = (A.T @ C_inv @ A) / (A.T @ C_inv @ Y)

# Make a plot
ti = np.linspace(0, 1, 1000)
Ai = Finufft1DRealOperator(ti, n_modes=K)

# Make a plot
fig, ax = plt.subplots()
ax.errorbar(t, Y, yerr=y_err, c="k", fmt="o")
ax.plot(ti, Ai @ θ)
ax.set_xlabel("t")
ax.set_ylabel("y")
```
![figure2](https://github.com/user-attachments/assets/b4797d52-b0d4-42e6-942f-24e07304eb71)
