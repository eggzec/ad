"""
Mathematical operations that generalize many operations from the standard math
and cmath modules so that they also track first and second derivatives.

The basic philosophy of order of type-operations is this:
A. Is X from the ADF class or subclass?
   1. Yes - Perform automatic differentiation.
   2. No - Is X an array object?
      a. Yes - Vectorize the operation and repeat at A for each item.
      b. No - Let the math/cmath function deal with X since it's probably a
         base numeric type. Otherwise they will throw the respective
         exceptions.

(c) 2013 by Abraham Lee <tisimst@gmail.com>.
Please send feature requests, bug reports, or feedback to this address.

This software is released under a dual license.  (1) The BSD license.
(2) Any other license, as long as it is obtained from the original
author.

"""

import cmath
import importlib.util
import math
from collections.abc import Callable
from typing import TypeVar

from ad import ADF, _apply_chain_rule, to_auto_diff


numpy_installed = importlib.util.find_spec("numpy") is not None


__all__ = [
    "acos",
    "acosh",
    "acot",
    "acoth",
    "acsc",
    "acsch",
    "asec",
    "asech",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "ceil",
    "cos",
    "cosh",
    "cot",
    "coth",
    "csc",
    "csch",
    "degrees",
    "e",
    "erf",
    "erfc",
    "exp",
    "expm1",
    "fabs",
    "factorial",
    "floor",
    "gamma",
    "hypot",
    "isinf",
    "isnan",
    "lgamma",
    "ln",
    "log",
    "log1p",
    "log10",
    "phase",
    "pi",
    "polar",
    "power",
    "radians",
    "rect",
    "sec",
    "sech",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
]


# FUNCTIONS IN THE MATH MODULE ##############################################
#
# Currently, there is no implementation for the following math module methods:
# - copysign
# - factorial <- depends on gamma
# - fmod
# - frexp
# - fsum
# - gamma* <- currently uses high-accuracy finite difference derivatives
# - lgamma <- depends on gamma
# - ldexp
# - modf
#
# we'll see if they(*) get implemented

e = math.e
pi = math.pi

eps = 1e-8

_HALF = 0.5
_PAIR = 2

F = TypeVar("F", bound=Callable[..., object])


def _fourth_order_first_fd(func: Callable[[float], float], x: float) -> float:
    """Compute the fourth-order accurate first finite-difference derivative.

    Parameters
    ----------
    func : Callable[[float], float]
        Scalar function to differentiate.
    x : float
        Point at which to evaluate the derivative.

    Returns
    -------
    float
        Fourth-order accurate finite-difference derivative of ``func`` at
        ``x``.
    """
    fm2 = func(x - 2 * eps)
    fm1 = func(x - eps)
    fp1 = func(x + eps)
    fp2 = func(x + 2 * eps)
    return (fm2 - 8 * fm1 + 8 * fp1 - fp2) / 12 / eps


def _fourth_order_second_fd(func: Callable[[float], float], x: float) -> float:
    """Compute the fourth-order accurate second finite-difference derivative.

    Parameters
    ----------
    func : Callable[[float], float]
        Scalar function to differentiate twice.
    x : float
        Point at which to evaluate the second derivative.

    Returns
    -------
    float
        Fourth-order accurate finite-difference second derivative of ``func``
        at ``x``.
    """
    fm2 = func(x - 2 * eps)
    fm1 = func(x - eps)
    f = func(x)
    fp1 = func(x + eps)
    fp2 = func(x + 2 * eps)
    return (-fm2 + 16 * fm1 - 30 * f + 16 * fp1 - fp2) / 12 / eps**2


def _vectorize(func: F) -> F:
    """
    Make a function that accepts 1 or 2 arguments work with input arrays.

    The supported input array length combinations are:

    - ``m x m``
    - ``1 x m``
    - ``m x 1``
    - ``1 x 1``

    Parameters
    ----------
    func : F
        Scalar function to vectorize.

    Returns
    -------
    F
        Wrapped version of ``func`` that broadcasts over array-like inputs.
    """

    def _apply_two_arg(x: object, y: object, **kwargs: object) -> object:
        try:
            return [
                vectorized_function(xi, yi, **kwargs)
                for xi, yi in zip(x, y, strict=False)
            ]
        except TypeError:
            pass
        try:
            return [vectorized_function(xi, y, **kwargs) for xi in x]
        except TypeError:
            pass
        try:
            return [vectorized_function(x, yi, **kwargs) for yi in y]
        except TypeError:
            return func(x, y, **kwargs)

    def vectorized_function(*args: object, **kwargs: object) -> object:
        if len(args) == 1:
            x = args[0]
            try:
                return [vectorized_function(xi, **kwargs) for xi in x]
            except TypeError:
                return func(x, **kwargs)

        if len(args) == _PAIR:
            x, y = args
            return _apply_two_arg(x, y, **kwargs)
        return None

    n = func.__name__
    m = func.__module__
    d = func.__doc__

    vectorized_function.__name__ = n
    vectorized_function.__module__ = m
    doc = f"Vectorized {n} function\n"
    if d is not None:
        doc += d
    vectorized_function.__doc__ = doc

    return vectorized_function


@_vectorize
def acos(x: float) -> float:
    """
    Return the arc cosine of x, in radians.

    Returns
    -------
    float
        Arc cosine of ``x`` in radians (or an :class:`ad.ADF` if ``x`` is
        one).
    """
    if isinstance(x, ADF):
        ad_funcs = list(map(to_auto_diff, [x]))

        x = ad_funcs[0].x

        f = acos(x)

        variables = ad_funcs[0]._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        lc_wrt_args = [-1 / sqrt(1 - x**2)]
        qc_wrt_args = [x / (sqrt(1 - x**2) * (x**2 - 1))]
        cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    if x.imag:
        return cmath.acos(x)
    return math.acos(x.real)


@_vectorize
def acosh(x: float) -> float:
    """
    Return the inverse hyperbolic cosine of x.

    Returns
    -------
    float
        Inverse hyperbolic cosine of ``x`` (or an :class:`ad.ADF` if ``x`` is
        one).
    """
    if isinstance(x, ADF):
        ad_funcs = list(map(to_auto_diff, [x]))

        x = ad_funcs[0].x

        f = acosh(x)

        variables = ad_funcs[0]._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        lc_wrt_args = [1 / sqrt(x**2 - 1)]
        qc_wrt_args = [-x / (x**2 - 1) ** 1.5]
        cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    if x.imag:
        return cmath.acosh(x)
    return math.acosh(x.real)


@_vectorize
def asin(x: float) -> float:
    """
    Return the arc sine of x, in radians.

    Returns
    -------
    float
        Arc sine of ``x`` in radians (or an :class:`ad.ADF` if ``x`` is one).
    """
    if isinstance(x, ADF):
        ad_funcs = list(map(to_auto_diff, [x]))

        x = ad_funcs[0].x

        f = asin(x)

        variables = ad_funcs[0]._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        lc_wrt_args = [1 / sqrt(1 - x**2)]
        qc_wrt_args = [-x / (sqrt(1 - x**2) * (x**2 - 1))]
        cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    if x.imag:
        return cmath.asin(x)
    return math.asin(x.real)


@_vectorize
def asinh(x: float) -> float:
    """
    Return the inverse hyperbolic sine of x.

    Returns
    -------
    float
        Inverse hyperbolic sine of ``x`` (or an :class:`ad.ADF` if ``x`` is
        one).
    """
    if isinstance(x, ADF):
        ad_funcs = list(map(to_auto_diff, [x]))

        x = ad_funcs[0].x

        f = asinh(x)

        variables = ad_funcs[0]._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        lc_wrt_args = [1 / sqrt(x**2 + 1)]
        qc_wrt_args = [-x / (x**2 + 1) ** 1.5]
        cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    if x.imag:
        return cmath.asinh(x)
    return math.asinh(x.real)


@_vectorize
def atan(x: float) -> float:
    """
    Return the arc tangent of x, in radians.

    Returns
    -------
    float
        Arc tangent of ``x`` in radians (or an :class:`ad.ADF` if ``x`` is
        one).
    """
    if isinstance(x, ADF):
        ad_funcs = list(map(to_auto_diff, [x]))

        x = ad_funcs[0].x

        f = atan(x)

        variables = ad_funcs[0]._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        lc_wrt_args = [1 / (x**2 + 1)]
        qc_wrt_args = [-2 * x / (x**4 + 2 * x**2 + 1)]
        cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    if x.imag:
        return cmath.atan(x)
    return math.atan(x.real)


@_vectorize
def atan2(y: float, x: float) -> float:
    """
    Return ``atan(y / x)``, in radians.

    The result is between ``-pi`` and ``pi``.

    Returns
    -------
    float
        Two-argument arc tangent of ``y`` and ``x`` in radians.
    """
    if x > 0:
        return atan(y / x)
    if x < 0:
        if y >= 0:
            return atan(y / x) + pi
        return atan(y / x) - pi
    if y > 0:
        return +pi / 2
    if y < 0:
        return -pi / 2
    return 0.0


@_vectorize
def atanh(x: float) -> float:
    """
    Return the inverse hyperbolic tangent of x.

    Returns
    -------
    float
        Inverse hyperbolic tangent of ``x`` (or an :class:`ad.ADF` if ``x``
        is one).
    """
    if isinstance(x, ADF):
        ad_funcs = list(map(to_auto_diff, [x]))

        x = ad_funcs[0].x

        f = atanh(x)

        variables = ad_funcs[0]._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        lc_wrt_args = [-1.0 / (x**2 - 1)]
        qc_wrt_args = [2 * x / (x**4 - 2 * x**2 + 1)]
        cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    if x.imag:
        return cmath.atanh(x)
    return math.atanh(x.real)


@_vectorize
def isinf(x: float) -> bool:
    """
    Return True if the real or imaginary part of x is infinite.

    Returns
    -------
    bool
        ``True`` if any part of ``x`` is positive or negative infinity.
    """
    if isinstance(x, ADF):
        return isinf(x.x)
    if x.imag:
        return cmath.isinf(x)
    return math.isinf(x.real)


@_vectorize
def isnan(x: float) -> bool:
    """
    Return True if the real or imaginary part of x is not a number (NaN).

    Returns
    -------
    bool
        ``True`` if any part of ``x`` is NaN.
    """
    if isinstance(x, ADF):
        return isnan(x.x)
    if x.imag:
        return cmath.isnan(x)
    return math.isnan(x.real)


@_vectorize
def phase(x: complex) -> float:
    """
    Return the phase of x (also known as the argument of x).

    Returns
    -------
    float
        Argument of ``x`` in radians.
    """
    return atan2(x.imag, x.real)


@_vectorize
def polar(x: complex) -> tuple[float, float]:
    """
    Return the representation of x in polar coordinates.

    Returns
    -------
    tuple[float, float]
        Pair ``(r, phi)`` of magnitude and phase.
    """
    return (abs(x), phase(x))


@_vectorize
def rect(r: float, phi: float) -> complex:
    """
    Return the complex number x with polar coordinates r and phi.

    Returns
    -------
    complex
        Complex value at the given polar coordinates.
    """
    return r * (cos(phi) + sin(phi) * 1j)


@_vectorize
def ceil(x: float) -> float:
    """
    Return the ceiling of x as a float.

    Returns
    -------
    float
        Smallest integer value greater than or equal to ``x``.
    """
    if isinstance(x, ADF):
        ad_funcs = list(map(to_auto_diff, [x]))

        x = ad_funcs[0].x

        f = ceil(x)

        variables = ad_funcs[0]._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        lc_wrt_args = [0.0]
        qc_wrt_args = [0.0]
        cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    return math.ceil(x)


@_vectorize
def cos(x: float) -> float:
    """
    Return the cosine of x radians.

    Returns
    -------
    float
        Cosine of ``x`` (or an :class:`ad.ADF` if ``x`` is one).
    """
    if isinstance(x, ADF):
        ad_funcs = list(map(to_auto_diff, [x]))

        x = ad_funcs[0].x

        f = cos(x)

        variables = ad_funcs[0]._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        lc_wrt_args = [-sin(x)]
        qc_wrt_args = [-cos(x)]
        cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    if x.imag:
        return cmath.cos(x)
    return math.cos(x.real)


@_vectorize
def cosh(x: float) -> float:
    """
    Return the hyperbolic cosine of x.

    Returns
    -------
    float
        Hyperbolic cosine of ``x`` (or an :class:`ad.ADF` if ``x`` is one).
    """
    if isinstance(x, ADF):
        ad_funcs = list(map(to_auto_diff, [x]))

        x = ad_funcs[0].x

        f = cosh(x)

        variables = ad_funcs[0]._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        lc_wrt_args = [sinh(x)]
        qc_wrt_args = [cosh(x)]
        cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    if x.imag:
        return cmath.cosh(x)
    return math.cosh(x.real)


@_vectorize
def degrees(x: float) -> float:
    """
    Convert angle ``x`` from radians to degrees.

    Returns
    -------
    float
        Angle ``x`` expressed in degrees.
    """
    return (180 / pi) * x


@_vectorize
def erf(x: float) -> float:
    """
    Return the error function at x.

    Returns
    -------
    float
        Error function evaluated at ``x``.
    """
    if isinstance(x, ADF):
        ad_funcs = list(map(to_auto_diff, [x]))

        x = ad_funcs[0].x

        f = erf(x)

        variables = ad_funcs[0]._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        lc_wrt_args = [2 * exp(-(x**2)) / sqrt(pi)]
        qc_wrt_args = [-4 * x * exp(-(x**2)) / sqrt(pi)]
        cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    return math.erf(x)


@_vectorize
def erfc(x: float) -> float:
    """
    Return the complementary error function at x.

    Returns
    -------
    float
        Complementary error function evaluated at ``x``.
    """
    return 1 - erf(x)


@_vectorize
def exp(x: float) -> float:
    """
    Return the exponential value of x.

    Returns
    -------
    float
        Exponential ``e**x`` (or an :class:`ad.ADF` if ``x`` is one).
    """
    if isinstance(x, ADF):
        ad_funcs = list(map(to_auto_diff, [x]))

        x = ad_funcs[0].x

        f = exp(x)

        variables = ad_funcs[0]._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        lc_wrt_args = [exp(x)]
        qc_wrt_args = [exp(x)]
        cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    if x.imag:
        return cmath.exp(x)
    return math.exp(x.real)


@_vectorize
def expm1(x: float) -> float:
    """
    Return ``e**x - 1``.

    For small floats ``x``, the subtraction in ``exp(x) - 1`` can result in a
    significant loss of precision; the ``expm1()`` function provides a way to
    compute this quantity to full precision.

    Returns
    -------
    float
        Value of ``e**x - 1`` computed to full precision.
    """
    if isinstance(x, ADF):
        ad_funcs = list(map(to_auto_diff, [x]))

        x = ad_funcs[0].x

        f = expm1(x)

        variables = ad_funcs[0]._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        lc_wrt_args = [exp(x)]
        qc_wrt_args = [exp(x)]
        cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    return math.expm1(x)


@_vectorize
def fabs(x: float) -> float:
    """
    Return the absolute value of x.

    Returns
    -------
    float
        Absolute value of ``x`` (or an :class:`ad.ADF` if ``x`` is one).
    """
    if isinstance(x, ADF):
        ad_funcs = list(map(to_auto_diff, [x]))

        x = ad_funcs[0].x

        f = fabs(x)

        variables = ad_funcs[0]._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        try:
            lc_wrt_args = [x / fabs(x)]
            qc_wrt_args = [1 / fabs(x) - (x**2) / fabs(x) ** 3]
        except ZeroDivisionError:
            lc_wrt_args = [0.0]
            qc_wrt_args = [0.0]

        cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    return math.fabs(x)


@_vectorize
def factorial(x: float) -> float:
    """
    Return ``x`` factorial.

    Uses the relationship ``factorial(x) == gamma(x + 1)`` to calculate
    derivatives.

    Returns
    -------
    float
        Factorial of ``x``.
    """
    return gamma(x + 1)


@_vectorize
def floor(x: float) -> float:
    """
    Return the floor of x as a float.

    Returns
    -------
    float
        Largest integer value less than or equal to ``x``.
    """
    if isinstance(x, ADF):
        ad_funcs = list(map(to_auto_diff, [x]))

        x = ad_funcs[0].x

        f = floor(x)

        variables = ad_funcs[0]._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        lc_wrt_args = [0.0]
        qc_wrt_args = [0.0]
        cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    return math.floor(x)


@_vectorize
def gamma(x: float) -> float:
    """
    Return the Gamma function at x (uses the Lanczos approximation).

    Returns
    -------
    float
        Gamma function evaluated at ``x``.
    """
    g = 607.0 / 128
    p = [
        0.99999999999999709182,
        57.156235665862923517,
        -59.597960355475491248,
        14.136097974741747174,
        -0.49191381609762019978,
        0.33994649984811888699e-4,
        0.46523628927048575665e-4,
        -0.98374475304879564677e-4,
        0.15808870322491248884e-3,
        -0.21026444172410488319e-3,
        0.21743961811521264320e-3,
        -0.16431810653676389022e-3,
        0.84418223983852743293e-4,
        -0.26190838401581408670e-4,
        0.36899182659531622704e-5,
    ]

    if x.real < _HALF:
        return pi / (sin(pi * x) * gamma(1 - x))
    x -= 1
    z = p[0]
    for i in range(1, len(p)):
        z += p[i] / (x + i)
    t = x + g + 0.5
    gamma_ = sqrt(2 * pi) * t ** (x + 0.5) * exp(-t) * z
    if isinstance(x, ADF):
        if gamma_.imag:
            return gamma_
        return gamma_.to_float()
    if gamma_.imag:
        return gamma_
    return gamma_.real


@_vectorize
def hypot(x: float, y: float) -> float:
    """
    Return the Euclidean norm, ``sqrt(x*x + y*y)``.

    Returns
    -------
    float
        Length of the vector from the origin to point ``(x, y)``.
    """
    return sqrt(x * x + y * y)


@_vectorize
def lgamma(x: float) -> float:
    """
    Return the natural logarithm of the absolute value of Gamma at ``x``.

    Returns
    -------
    float
        ``log(abs(gamma(x)))`` evaluated at ``x``.
    """
    return log(abs(gamma(x)))


@_vectorize
def log(x: float, base: float | None = None) -> float:
    """
    Return the logarithm of ``x``.

    With one argument, return the natural logarithm of ``x`` (to base ``e``).
    With two arguments, return the logarithm of ``x`` to the given base,
    calculated as ``log(x)/log(base)``.

    Returns
    -------
    float
        Logarithm of ``x`` (in the requested base, when given).
    """
    if base is None:
        return log(x, base=e)

    if isinstance(x, ADF):
        ad_funcs = list(map(to_auto_diff, [x]))

        x = ad_funcs[0].x

        f = log(x, base)

        variables = ad_funcs[0]._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        lc_wrt_args = [1.0 / (x * ln(base))]
        qc_wrt_args = [-1.0 / (x**2 * ln(base))]
        cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)

    if x.imag:
        return cmath.log(x, base)
    return math.log(x.real, base)


@_vectorize
def log10(x: float) -> float:
    """
    Return the base-10 logarithm of x.

    This is usually more accurate than ``log(x, 10)``.

    Returns
    -------
    float
        Base-10 logarithm of ``x``.
    """
    if isinstance(x, ADF):
        ad_funcs = list(map(to_auto_diff, [x]))

        x = ad_funcs[0].x

        f = log10(x)

        variables = ad_funcs[0]._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        lc_wrt_args = [1 / x / log(10)]
        qc_wrt_args = [-1.0 / x**2 / log(10)]
        cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)

    if x.imag:
        return cmath.log10(x)
    return math.log10(x.real)


@_vectorize
def log1p(x: float) -> float:
    """
    Return the natural logarithm of ``1 + x``.

    Returns
    -------
    float
        ``log(1 + x)`` evaluated to full precision for small ``x``.
    """
    if isinstance(x, ADF):
        ad_funcs = list(map(to_auto_diff, [x]))

        x = ad_funcs[0].x

        f = log1p(x)

        variables = ad_funcs[0]._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        lc_wrt_args = [1 / (x + 1)]
        qc_wrt_args = [-1.0 / (x + 1) ** 2]
        cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)

    return math.log1p(x)


@_vectorize
def power(x: float, y: float) -> float:
    """
    Return ``x`` raised to the power ``y``.

    Returns
    -------
    float
        ``x ** y``.
    """
    return x**y


@_vectorize
def radians(x: float) -> float:
    """
    Convert angle ``x`` from degrees to radians.

    Returns
    -------
    float
        Angle ``x`` expressed in radians.
    """
    return (pi / 180) * x


@_vectorize
def sin(x: float) -> float:
    """
    Return the sine of ``x`` in radians.

    Returns
    -------
    float
        Sine of ``x`` (or an :class:`ad.ADF` if ``x`` is one).
    """
    if isinstance(x, ADF):
        ad_funcs = list(map(to_auto_diff, [x]))

        x = ad_funcs[0].x

        f = sin(x)

        variables = ad_funcs[0]._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        lc_wrt_args = [cos(x)]
        qc_wrt_args = [-sin(x)]
        cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    if x.imag:
        return cmath.sin(x)
    return math.sin(x.real)


@_vectorize
def sinh(x: float) -> float:
    """
    Return the hyperbolic sine of ``x``.

    Returns
    -------
    float
        Hyperbolic sine of ``x`` (or an :class:`ad.ADF` if ``x`` is one).
    """
    if isinstance(x, ADF):
        ad_funcs = list(map(to_auto_diff, [x]))

        x = ad_funcs[0].x

        f = sinh(x)

        variables = ad_funcs[0]._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        lc_wrt_args = [cosh(x)]
        qc_wrt_args = [sinh(x)]
        cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    if x.imag:
        return cmath.sinh(x)
    return math.sinh(x.real)


@_vectorize
def sqrt(x: float) -> float:
    """
    Return the square root of ``x``.

    Returns
    -------
    float
        Square root of ``x``.
    """
    return x**0.5


@_vectorize
def tan(x: float) -> float:
    """
    Return the tangent of ``x`` in radians.

    Returns
    -------
    float
        Tangent of ``x`` (or an :class:`ad.ADF` if ``x`` is one).
    """
    if isinstance(x, ADF):
        ad_funcs = list(map(to_auto_diff, [x]))

        x = ad_funcs[0].x

        f = tan(x)

        variables = ad_funcs[0]._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        lc_wrt_args = [1.0 / (cos(x)) ** 2]
        qc_wrt_args = [2 * sin(x) / (cos(x)) ** 3]
        cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    if x.imag:
        return cmath.tan(x)
    return math.tan(x.real)


@_vectorize
def tanh(x: float) -> float:
    """
    Return the hyperbolic tangent of ``x``.

    Returns
    -------
    float
        Hyperbolic tangent of ``x`` (or an :class:`ad.ADF` if ``x`` is one).
    """
    if isinstance(x, ADF):
        ad_funcs = list(map(to_auto_diff, [x]))

        x = ad_funcs[0].x

        f = tanh(x)

        variables = ad_funcs[0]._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        lc_wrt_args = [1.0 / (cosh(x)) ** 2]
        qc_wrt_args = [-2 * sinh(x) / (cosh(x)) ** 3]
        cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    if x.imag:
        return cmath.tanh(x)
    return math.tanh(x.real)


@_vectorize
def trunc(x: float) -> float:
    """
    Return the **Real** value ``x`` truncated to an **Integral**.

    Uses the ``__trunc__`` method.

    Returns
    -------
    float
        Truncated integer value of ``x``.
    """
    if isinstance(x, ADF):
        ad_funcs = list(map(to_auto_diff, [x]))

        x = ad_funcs[0].x

        f = trunc(x)

        variables = ad_funcs[0]._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        lc_wrt_args = [0.0]
        qc_wrt_args = [0.0]
        cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    try:
        return [trunc(xi) for xi in x]
    except TypeError:
        return math.trunc(x)


# OTHER CONVENIENCE FUNCTIONS ###############################################


@_vectorize
def csc(x: float) -> float:
    """
    Return the cosecant of ``x``.

    Returns
    -------
    float
        Cosecant ``1 / sin(x)``.
    """
    return 1.0 / sin(x)


@_vectorize
def sec(x: float) -> float:
    """
    Return the secant of ``x``.

    Returns
    -------
    float
        Secant ``1 / cos(x)``.
    """
    return 1.0 / cos(x)


@_vectorize
def cot(x: float) -> float:
    """
    Return the cotangent of ``x``.

    Returns
    -------
    float
        Cotangent ``1 / tan(x)``.
    """
    return 1.0 / tan(x)


@_vectorize
def csch(x: float) -> float:
    """
    Return the hyperbolic cosecant of ``x``.

    Returns
    -------
    float
        Hyperbolic cosecant ``1 / sinh(x)``.
    """
    return 1.0 / sinh(x)


@_vectorize
def sech(x: float) -> float:
    """
    Return the hyperbolic secant of ``x``.

    Returns
    -------
    float
        Hyperbolic secant ``1 / cosh(x)``.
    """
    return 1.0 / cosh(x)


@_vectorize
def coth(x: float) -> float:
    """
    Return the hyperbolic cotangent of ``x``.

    Returns
    -------
    float
        Hyperbolic cotangent ``1 / tanh(x)``.
    """
    return 1.0 / tanh(x)


@_vectorize
def acsc(x: float) -> float:
    """
    Return the inverse cosecant of ``x``.

    Returns
    -------
    float
        Inverse cosecant ``asin(1 / x)``.
    """
    return asin(1.0 / x)


@_vectorize
def asec(x: float) -> float:
    """
    Return the inverse secant of ``x``.

    Returns
    -------
    float
        Inverse secant ``acos(1 / x)``.
    """
    return acos(1.0 / x)


@_vectorize
def acot(x: float) -> float:
    """
    Return the inverse cotangent of ``x``.

    Returns
    -------
    float
        Inverse cotangent ``atan(1 / x)``.
    """
    return atan(1.0 / x)


@_vectorize
def acsch(x: float) -> float:
    """
    Return the inverse hyperbolic cosecant of ``x``.

    Returns
    -------
    float
        Inverse hyperbolic cosecant ``asinh(1 / x)``.
    """
    return asinh(1.0 / x)


@_vectorize
def asech(x: float) -> float:
    """
    Return the inverse hyperbolic secant of ``x``.

    Returns
    -------
    float
        Inverse hyperbolic secant ``acosh(1 / x)``.
    """
    return acosh(1.0 / x)


@_vectorize
def acoth(x: float) -> float:
    """
    Return the inverse hyperbolic cotangent of ``x``.

    Returns
    -------
    float
        Inverse hyperbolic cotangent ``atanh(1 / x)``.
    """
    return atanh(1.0 / x)


@_vectorize
def ln(x: float) -> float:
    """
    Return the natural logarithm of ``x``.

    Returns
    -------
    float
        Natural logarithm of ``x``.
    """
    return log(x)
