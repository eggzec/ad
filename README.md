# ad

**Fast, transparent first- and second-order automatic differentiation for Python**

[![Tests](https://github.com/eggzec/ad/actions/workflows/code_test.yml/badge.svg)](https://github.com/eggzec/ad/actions/workflows/code_test.yml)
[![Documentation](https://github.com/eggzec/ad/actions/workflows/docs_build.yml/badge.svg)](https://github.com/eggzec/ad/actions/workflows/docs_build.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![codecov](https://codecov.io/github/eggzec/ad/graph/badge.svg)](https://codecov.io/github/eggzec/ad)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=eggzec_ad&metric=alert_status)](https://sonarcloud.io/project/overview?id=eggzec_ad)
[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-blue.svg)](LICENSE)

[![PyPI Downloads](https://img.shields.io/pypi/dm/ad.svg?label=PyPI%20downloads)](https://pypi.org/project/ad/)
[![Python versions](https://img.shields.io/pypi/pyversions/ad.svg)](https://pypi.org/project/ad/)

## Overview

The `ad` package allows you to easily and transparently perform first- and
second-order automatic differentiation. Advanced math involving trigonometric,
logarithmic, hyperbolic, and related functions can be evaluated directly using
the `ad.admath` submodule.

All base numeric types are supported (`int`, `float`, `complex`, etc.). The
package is designed so underlying numeric types interact as they normally do
during calculations. In practice, `ad` behaves like a lightweight wrapper that
tracks derivatives while preserving standard numeric behavior.

From the Wikipedia entry on
[Automatic differentiation](http://en.wikipedia.org/wiki/Automatic_differentiation):

> "AD exploits the fact that every computer program, no matter how
> complicated, executes a sequence of elementary arithmetic operations
> (addition, subtraction, multiplication, division, etc.) and elementary
> functions (exp, log, sin, cos, etc.). By applying the chain rule repeatedly
> to these operations, derivatives of arbitrary order can be computed
> automatically, and accurate to working precision."

See the
[package documentation](http://pythonhosted.org/ad)
for details and examples.

## Main Features

- Transparent calculations with derivatives, requiring little or no
  modification to existing code (including NumPy-based code).
- Broad mathematical operation support, including most functions from
  [`math`](http://docs.python.org/library/math.html) and
  [`cmath`](http://docs.python.org/library/cmath.html), plus convenience
  trigonometric, hyperbolic, and logarithmic helpers (`csc`, `acoth`, `ln`,
  etc.). Comparison operators follow the same rules as the wrapped numeric
  values.
- Seamless real and complex arithmetic through `ad.admath` counterparts.
- Automatic gradient and Hessian function generator for optimization workflows
  with `scipy.optimize` via `gh(your_func_here)`.
- Linear algebra routines in `ad.linalg` similar to NumPy's `linalg`, without
  LAPACK dependency.

### Linear Algebra Routines

**Decompositions**

- `chol`: Cholesky decomposition
- `lu`: LU decomposition
- `qr`: QR decomposition

**Solving equations and matrix inversion**

- `solve`: General solver for linear systems
- `lstsq`: Least-squares solver for linear systems
- `inv`: Multiplicative inverse of a matrix


## Installation

```bash
uv pip install ad
```

Requires Python 3.10+. See the
[full installation guide](https://eggzec.github.io/ad/installation/).

## Documentation

- [Theory](https://eggzec.github.io/ad/theory/) — mathematical background, hierarchical basis, algorithms
- [Quickstart](https://eggzec.github.io/ad/quickstart/) — runnable examples
- [API Reference](https://eggzec.github.io/ad/api/) — class and function signatures
- [References](https://eggzec.github.io/ad/references/) — literature citations

## License

BSD-3-Clause.
