# Quickstart

## An easy-to-use calculator

Calculations involving **differentiation** can be performed even without knowing anything about the Python programming language. After installing this package and invoking the Python interpreter, calculations with **automatic differentiation** can be performed **transparently** (i.e., through the usual syntax for mathematical formulas):

```python
from ad import adnumber
from ad.admath import *  # sin(), etc.

x = adnumber(1)
print(2 * x)
# ad(2)
sin(2 * x)
# ad(0.9092974268256817)
```

So far, there should not be anything unexpected, but first and second derivatives can now be accessed through **intuitive methods**:

```python
y = sin(2 * x)
y.d(x)  # dy/dx at x=1
# -0.8322936730942848
y.d2(x)  # d2y/dx2 at x=1
# -3.637189707302727
```

Thus, existing calculation code designed for regular numbers can run with numbers that track derivatives with no or little modification.

## Basic setup

Basic mathematical operations involving numbers that track derivatives only require a simple import:

```python
from ad import adnumber
```

The `adnumber` function creates numbers with derivative tracing capabilities. Existing calculation code can usually run with no or little modification and automatically produce derivatives.

The `ad` module contains other features, which can be made accessible through:

```python
import ad
```

## Creating automatic differentiation numbers

Numbers that track their derivatives are input just as you would for any normal numeric type. In that sense, they are basically wrapped without really changing their fundamental type.

```python
x = adnumber(2)  # acts like an int object
x = adnumber(2.0)  # acts like a float object
x = adnumber(2 + 0j)  # acts like a complex object
```

Mathematical calculations that follow are interpreted based upon the base numeric types involved.

## Basic math

Calculations can be performed directly, as with regular real or complex numbers:

```python
square = x**2
print(square)
# ad(4)
a = adnumber(3 + 4j)
print(a)
# ad((3+4j))
abs(a)
# ad(5.0)
b = adnumber(1 - 1j)
a * b
# ad((7+1j))
a.real, a.imag
# (3.0, 4.0)
```

AD objects that represent real values can also be used to create complex ones:

```python
y = adnumber(3.14)
z = x + y * 1j
print(z)
# ad((2+3.14j))
```

If an AD object is used as input to `adnumber`, then a deepcopy is made, but no tracking relation is created between the input and output objects:

```python
z = adnumber(x)
z is x
# False
z == x
# True
z.d(x)
# 0.0
z.d(z)
# 1.0
x.d(z)
# 0.0
```

## Mathematical operations

Besides being able to apply basic mathematical operations, this package provides generalizations of **most of the functions from the standard** `math` **and** `cmath` **modules**.

These mathematical functions are found in the `ad.admath` module:

```python
from ad.admath import *  # Imports sin(), etc.

sin(x**2)
# ad(-0.7568024953079282)
```

There are also many other functions not normally found in the `math` and `cmath` modules that are conveniently available, like `csc` and others.

The list of available mathematical functions can be obtained with:

```bash
pydoc ad.admath
```

## Arrays of numbers

It is possible to put automatic differentiation numbers within NumPy arrays and matrices, lists, or tuples, and the returned object is of that respective type (even nested objects work):

```python
adnumber([1, [2, 3]])
# [ad(1), [ad(2), ad(3)]]
adnumber((1, 2))
# (ad(1), ad(2))
arr = adnumber(np.array([[1, 2], [3, 4]]))
2 * arr
# array([[ad(2), ad(4)],
#        [ad(6), ad(8)]], dtype=object)
arr.sum()
# ad(10)
```

Thus, usual operations on NumPy arrays can be performed transparently even when these arrays contain numbers that track derivatives.

## Access to the derivatives and to the nominal value

The nominal value and the derivatives can be accessed independently:

```python
print(square)
# ad(4)
print(square.x)
# 4
print(square.d(x))
# 4.0
print(square.d2(x))
# 2.0
print(square.d())
# {ad(4): 4.0}
y = adnumber(1.5)
print(square.d(y))
# 0.0
z = square / y
z.d2c(x, y)
# -1.7777777777777777
z.d(square)
# 0.0
```

## Access to more than one derivative

Arrays of derivatives can be obtained through the `gradient` and `hessian` methods.

```python
u = adnumber(0.1, "u")
v = adnumber(3.14, "v")

sum_value = u + 2 * v / u
sum_value.d()
# {ad(0.1, u): -626.9999999999999, ad(3.14, v): 20.0}

sum_value.gradient([u, v])
# [-626.9999999999999, 20.0]

sum_value.hessian([u, v])
# [[12559.999999999998, -199.99999999999997], [-199.99999999999997, 0.0]]

from ad import jacobian

jacobian([square, sum_value], [x, u, v])
# [[4.0, 0.0, 0.0], [0.0, -626.9999999999999, 20.0]]
```

## Comparison operators

Comparison operators behave naturally as they would with numbers outside of this package, even with other scalar values:

```python
x = adnumber(0.2)
y = adnumber(1)
y > x
# True
y > 0
# True
y == 1.0
# True
```

## Making custom functions accept numbers that track derivatives

Due to the nature of automatic differentiation, unless a function can be represented with a mathematical equation, automatic differentiation is meaningless. For custom functions that cannot be represented mathematically (that is, those that do not have an analytical form), derivatives may be calculated using other means like finite-difference derivatives.

## Miscellaneous utilities

It is sometimes useful to use the gradients and Hessians provided by this package for the purpose of supplementing an optimization routine, like those in the `scipy.optimize` submodule.

With this package, a function can be conveniently wrapped with functions that return both the gradient and Hessian:

```python
from ad import gh


def my_cool_function(x):
    return (x[0] - 10.0) ** 2 + (x[1] + 5.0) ** 2


my_cool_gradient, my_cool_hessian = gh(my_cool_function)
```

These objects (`my_cool_gradient` and `my_cool_hessian`) are functions that accept an array `x` and other optional args. Depending on the optimization routine, you may be able to use only the gradient function:

```python
from scipy.optimize import minimize

x0 = [24, 17]
bnds = ((0, None), (0, None))
res = minimize(
    my_cool_function,
    x0,
    bounds=bnds,
    method="L-BFGS-B",
    jac=my_cool_gradient,
    options={"ftol": 1e-8, "disp": False},
)

res.x
# array([ 10.,   0.])
res.fun
# 25.0
res.jac
# array([  7.10542736e-15,   1.00000000e+01])
```

You might wonder why the final gradient (`res.jac`) is not precisely `[0, 10]`. It is not because of numerical error in the AD methods. If all digits are printed, the apparent exact point is not exactly `[10, 0]`:

```python
list(res.x)
# [10.000000000000004, 0.0]
```

The old docs note that with finite upper bounds in this setup, the exact answer is obtained:

```python
bnds = ((0, 100), (0, 100))
res = minimize(
    my_cool_function,
    x0,
    bounds=bnds,
    method="L-BFGS-B",
    jac=my_cool_gradient,
    options={"ftol": 1e-8, "disp": True},
)

list(res.x)
# [10.0, 0.0]
list(res.jac)
# [0.0, 10.0]
```

Notice that use of `gh` does not require explicitly initializing any variable with `adnumber` since this happens internally with the wrapped functions.
