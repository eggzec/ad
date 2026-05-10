# API Reference

## Core Imports

```python
from ad import adnumber
import ad
from ad.admath import *
from ad import jacobian, gh
```

## Core Factory

- `adnumber(value, tag=None)` creates numbers with derivative-tracing capabilities.

## Derivative Accessors

- `d(var=None)`: first derivative with respect to `var`.
- `d2(var=None)`: second derivative with respect to `var`.
- `d2c(var1, var2)`: second cross-derivative with respect to two variables.
- `gradient(vars)`: gradient vector with respect to variables.
- `hessian(vars)`: Hessian matrix with respect to variables.

## Nominal Value Access

- `x`: underlying numeric value stored in an AD object.

## Jacobian Helper

The **jacobian matrix** can be created for multiple dependent objects, where each row is the gradient of the dependent variables with respect to each of the independent variables, in the order specified:

```python
from ad import jacobian
jacobian([square, sum_value], [x, u, v])
```

## `ad.admath` Mathematical Functions

Besides basic operators, this package provides generalizations of most functions from the standard `math` and `cmath` modules.

Some functions, like `erf`, are only available in `math`, so an exception is raised if a complex number is passed.

There are also many convenience functions not normally found in `math` and `cmath`, like `csc` and others.

The list of available mathematical functions can be obtained with:

```bash
pydoc ad.admath
```

## `ad.linalg` Routines

- Decompositions: `chol`, `lu`, `qr`
- Solvers and inverse: `solve`, `lstsq`, `inv`

### `chol`

```python
A = [[25, 15, -5],
     [15, 18,  0],
     [-5,  0, 11]]

L = chol(A)
U = chol(A, 'upper')
```

### `lu`

```python
A = [[1, 3, 5],
     [2, 4, 7],
     [1, 1, 0]]

L, U, P = lu(A)
```

### `qr`

```python
A = [[12, -51,   4],
     [ 6, 167, -68],
     [-4,  24, -41]]

q, r = qr(A)
```

### `solve`

```python
A = [[1, 2, 1], [4, 6, 3], [9, 8, 2]]
b = [3, 2, 1]
solve(A, b)
```

### `lstsq`

```python
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([3, 6, 11, 18, 27, 38])
y = y + np.random.randn(len(y))
A = np.vstack([np.ones(len(x)), x, x**2]).T
b = lstsq(A, y)
```

### `inv`

```python
A = [[25, 15, -5],
     [15, 18,  0],
     [-5,  0, 11]]
Ainv = inv(A)
```

## Optimization Utility: `gh`

It is sometimes useful to use gradients and Hessians for optimization routines (for example, in `scipy.optimize`).

With this package, a function can be wrapped with functions that return both gradient and Hessian:

```python
from ad import gh

def my_cool_function(x):
    return (x[0] - 10.0)**2 + (x[1] + 5.0)**2

my_cool_gradient, my_cool_hessian = gh(my_cool_function)
```

These objects are functions that accept an array `x` and other optional args.

## Type Checks and Classes

The recommended way of testing whether `value` tracks derivatives handled by this module is:

```python
isinstance(value, ad.ADF)
```

Numbers with derivatives are represented through two different classes:

1. Class for independent variables: `ADV` (inherits from `ADF`)
2. Class for functions that depend on independent variables: `ADF`

The factory function `adnumber` creates variables and returns an `ADV` object:

```python
x = adnumber(0.1)
type(x)
# <class 'ad.ADV'>
```

Mathematical expressions involving numbers with derivatives generally return `ADF` objects:

```python
type(admath.sin(x))
# <class 'ad.ADF'>
```

Documentation for these classes is available in their Python docstrings, which can for instance be displayed through `pydoc`.

## Useful Modules

- `ad`: core classes (`ADF`, `ADV`), `adnumber`, utilities.
- `ad.admath`: math/cmath-compatible AD functions.
- `ad.linalg`: AD-compatible linear algebra algorithms.
