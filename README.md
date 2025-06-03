<div align="Center">

# nifty-solve

Fit very flexible linear models using Fourier bases without ever constructing the design matrix.

[![Test Status](https://github.com/andycasey/nifty-solve/actions/workflows/ci.yml/badge.svg)](https://github.com/andycasey/nifty-solve/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/andycasey/nifty-solve/badge.svg?branch=main&service=github)](https://coveralls.io/github/andycasey/nifty-solve?branch=main)

</div>

> **Warning**  
> There is full test coverage on the standard operators (1D, 2D, 3D) for all dimension permutations. There is no current test coverage on the JAX operators.

# Install

```
uv add nifty-solve
```

If you plan to use JAX operators, you will need to use:
```
uv add nifty-solve[jax]
```

If that fails:
```
git clone git@github.com:andycasey/nifty-solve.git
```