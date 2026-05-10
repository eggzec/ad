"""
Created on Thu Apr 11 12:52:09 2013

@author: tisimst
"""

import cmath
import copy
import math
from numbers import Number


try:
    import numpy

    numpy_installed = True
except ImportError:
    numpy_installed = False

__version_info__ = (1, 3, 2)
__version__ = ".".join(list(map(str, __version_info__)))

__author__ = "Abraham Lee"

__all__ = ["adnumber", "gh", "jacobian"]

CONSTANT_TYPES = Number


def to_auto_diff(x: object) -> "ADF":
    """
    Transform ``x`` into an automatically differentiated function (ADF).

    If ``x`` is already an ADF (or a subclass), it is returned unchanged.

    Raises an exception unless ``x`` belongs to some specific classes of
    objects that are known not to depend on ADF objects (which then cannot be
    considered as constants).

    Returns
    -------
    ADF
        ``x`` wrapped as an ADF (or returned as-is if already one).
    """
    if isinstance(x, ADF):
        return x

    if isinstance(x, CONSTANT_TYPES):
        return ADF(x, {}, {}, {})

    raise NotImplementedError(
        f"Automatic differentiation not yet supported for {type(x)} objects"
    )


def _apply_chain_rule(
    ad_funcs: list,
    variables: set,
    lc_wrt_args: list,
    qc_wrt_args: list,
    cp_wrt_args: float,
) -> tuple[dict, dict, dict]:
    """
    Apply the first and second-order chain rule.

    Calculates the derivatives with respect to original variables (i.e.,
    objects created with the ``adnumber(...)`` constructor).

    For reference:

    - ``lc`` refers to "linear coefficients" or first-order terms
    - ``qc`` refers to "quadratic coefficients" or pure second-order terms
    - ``cp`` refers to "cross-product" second-order terms

    Returns
    -------
    tuple[dict, dict, dict]
        Tuple ``(lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)`` of derivative maps.
    """
    num_funcs = len(ad_funcs)

    lc_wrt_vars = dict((var, 0.0) for var in variables)
    qc_wrt_vars = dict((var, 0.0) for var in variables)
    cp_wrt_vars = {}
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i < j:
                cp_wrt_vars[var1, var2] = 0.0

    for j, var1 in enumerate(variables):
        for k, var2 in enumerate(variables):
            for f, dh, d2h in zip(
                ad_funcs, lc_wrt_args, qc_wrt_args, strict=False
            ):
                if j == k:
                    fdv1 = f.d(var1)
                    lc_wrt_vars[var1] += dh * fdv1

                    qc_wrt_vars[var1] += dh * f.d2(var1) + d2h * fdv1**2

                elif j < k:
                    tmp = dh * f.d2c(var1, var2) + d2h * f.d(var1) * f.d(var2)
                    cp_wrt_vars[var1, var2] += tmp

            if j == k and num_funcs > 1:
                tmp = (
                    2 * cp_wrt_args * ad_funcs[0].d(var1) * ad_funcs[1].d(var1)
                )
                qc_wrt_vars[var1] += tmp

            elif j < k and num_funcs > 1:
                tmp = cp_wrt_args * (
                    ad_funcs[0].d(var1) * ad_funcs[1].d(var2)
                    + ad_funcs[0].d(var2) * ad_funcs[1].d(var1)
                )
                cp_wrt_vars[var1, var2] += tmp

    return (lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)


def _floor(x: object) -> float:
    """
    Return the floor of ``x`` as a float.

    The largest integer value less than or equal to ``x``. This is required
    for the ``mod`` function.

    Returns
    -------
    float
        Floor value of ``x`` (or an :class:`ADF` when ``x`` is one).
    """
    if isinstance(x, ADF):
        ad_funcs = [to_auto_diff(x)]

        x = ad_funcs[0].x

        f = _floor(x)

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


class _ADFArithmetic:
    """
    Mixin providing the arithmetic dunder methods for :class:`ADF`.

    Splitting these into a mixin keeps the :class:`ADF` class itself within
    the public-method limit while preserving the public API.
    """

    def __add__(self, val: object) -> "ADF":
        ad_funcs = [self, to_auto_diff(val)]

        x = ad_funcs[0].x
        y = ad_funcs[1].x

        f = x + y

        variables = self._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        lc_wrt_args = [1.0, 1.0]
        qc_wrt_args = [0.0, 0.0]
        cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)

    def __radd__(self, val: object) -> "ADF":
        """Return ``val + self``.

        Returns
        -------
        ADF
            Result of ``val + self``.
        """
        return self + val

    def __mul__(self, val: object) -> "ADF":
        ad_funcs = [self, to_auto_diff(val)]

        x = ad_funcs[0].x
        y = ad_funcs[1].x

        f = x * y

        variables = self._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        lc_wrt_args = [y, x]
        qc_wrt_args = [0.0, 0.0]
        cp_wrt_args = 1.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)

    def __rmul__(self, val: object) -> "ADF":
        """Return ``val * self``.

        Returns
        -------
        ADF
            Result of ``val * self``.
        """
        return self * val

    def __truediv__(self, val: object) -> "ADF":
        ad_funcs = [self, to_auto_diff(val)]

        x = ad_funcs[0].x
        y = ad_funcs[1].x

        f = x / y

        variables = self._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        lc_wrt_args = [1.0 / y, -x / y**2]
        qc_wrt_args = [0.0, 2 * x / y**3]
        cp_wrt_args = -1.0 / y**2

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)

    def __rtruediv__(self, val: object) -> "ADF":
        """Return ``val / self``.

        Returns
        -------
        ADF
            Result of ``val * self ** -1``.
        """
        return val * self ** (-1)

    def __sub__(self, val: object) -> "ADF":
        """Return ``self - val``.

        Returns
        -------
        ADF
            Result of ``self + (-1 * val)``.
        """
        return self + (-1 * val)

    def __rsub__(self, val: object) -> "ADF":
        """Return ``val - self``.

        Returns
        -------
        ADF
            Result of ``-1 * self + val``.
        """
        return -1 * self + val

    def __pow__(self, val: object) -> "ADF":
        ad_funcs = [self, to_auto_diff(val)]

        x = ad_funcs[0].x
        y = ad_funcs[1].x

        f = x**y

        variables = self._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        if x.imag or y.imag:
            if abs(x) > 0 and ad_funcs[1].d(ad_funcs[1]) != 0:
                lc_wrt_args = [y * x ** (y - 1), x**y * cmath.log(x)]
                qc_wrt_args = [
                    y * (y - 1) * x ** (y - 2),
                    x**y * (cmath.log(x)) ** 2,
                ]
                cp_wrt_args = x ** (y - 1) * (y * cmath.log(x) + 1) / x
            else:
                lc_wrt_args = [y * x ** (y - 1), 0.0]
                qc_wrt_args = [y * (y - 1) * x ** (y - 2), 0.0]
                cp_wrt_args = 0.0
        else:
            x = x.real
            y = y.real
            if x > 0:
                lc_wrt_args = [y * x ** (y - 1), x**y * math.log(x)]
                qc_wrt_args = [
                    y * (y - 1) * x ** (y - 2),
                    x**y * (math.log(x)) ** 2,
                ]
                cp_wrt_args = x**y * (y * math.log(x) + 1) / x
            else:
                lc_wrt_args = [y * x ** (y - 1), 0.0]
                qc_wrt_args = [y * (y - 1) * x ** (y - 2), 0.0]
                cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)

    def __rpow__(self, val: object) -> "ADF":
        return to_auto_diff(val) ** self

    def __mod__(self, val: object) -> "ADF":
        return self - val * _floor(self / val)

    def __rmod__(self, val: object) -> "ADF":
        return val - self * _floor(val / self)

    def __neg__(self) -> "ADF":
        return -1 * self

    def __pos__(self) -> "ADF":
        return self

    def __invert__(self) -> "ADF":
        return -(self + 1)

    def __abs__(self) -> "ADF":
        ad_funcs = [self]

        x = ad_funcs[0].x

        f = abs(x)

        variables = self._get_variables(ad_funcs)

        if not variables or isinstance(f, bool):
            return f

        try:
            lc_wrt_args = [x / abs(x)]
        except ZeroDivisionError:
            lc_wrt_args = [0.0]

        qc_wrt_args = [0.0]
        cp_wrt_args = 0.0

        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
            ad_funcs, variables, lc_wrt_args, qc_wrt_args, cp_wrt_args
        )

        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)


class _ADFComparison:
    """
    Mixin providing the comparison dunder methods for :class:`ADF`.
    """

    __hash__ = object.__hash__

    def __eq__(self, val: object) -> bool:
        ad_funcs = [self, to_auto_diff(val)]
        return ad_funcs[0].x == ad_funcs[1].x

    def __ne__(self, val: object) -> bool:
        return not self == val

    def __lt__(self, val: object) -> bool:
        ad_funcs = [self, to_auto_diff(val)]
        return ad_funcs[0].x < ad_funcs[1].x

    def __le__(self, val: object) -> bool:
        return (self < val) or (self == val)

    def __gt__(self, val: object) -> bool:
        return not self <= val

    def __ge__(self, val: object) -> bool:
        return (self > val) or (self == val)


class ADF(_ADFArithmetic, _ADFComparison):
    """
    Automatically Differentiated Function (ADF).

    The ADF class contains derivative information about the results of a
    previous operation on any two objects where at least one is an ADF or
    ADV object.

    An ADF object has class members ``_lc``, ``_qc``, and ``_cp`` to contain
    first-order derivatives, second-order derivatives, and cross-product
    derivatives, respectively, of all ADV objects in the ADF's lineage. When
    requesting a cross-product term, either order of objects may be used,
    since mathematically they are equivalent.

    For example, if ``z = z(x, y)``, then::

          2       2
         d z     d z
        ----- = -----
        dx dy   dy dx


    Example
    -------
    Initialize some ADV objects (tag not required, but useful)::

        >>> x = adnumber(1, tag='x')
        >>> y = adnumber(2, tag='y')

    Now some basic math, showing the derivatives of the final result::

        >>> z = x + y
        >>> z.d()
        {ad(1.0, x): 1.0, ad(2.0, y): 1.0}
        >>> z.d2()
        {ad(1.0, x): 0.0, ad(2.0, y): 0.0}
        >>> z.d2c()
        {(ad(1.0, x), ad(2.0, y)): 0.0}

    """

    def __init__(
        self,
        value: object,
        lc: dict,
        qc: dict,
        cp: dict,
        tag: str | None = None,
    ) -> None:
        self.x = value
        self._lc = lc
        self._qc = qc
        self._cp = cp
        self.tag = tag

    def __hash__(self) -> int:
        return id(self)

    def trace_me(self) -> None:
        """
        Make this object traceable in future derivative calculations.

        The change is not retroactive.

        Caution
        -------
        When using ADF (i.e. dependent variable) objects as input to the
        derivative class methods, the returning value may only be useful with
        the ``d(...)`` and ``d2(...)`` methods.

        DO NOT MIX ADV AND ADF OBJECTS AS INPUTS TO THE ``d2c(...)`` METHOD
        SINCE THE RESULT IS NOT LIKELY TO BE NUMERICALLY MEANINGFUL.

        Example
        -------
        ::

            >>> x = adnumber(2.1)
            >>> y = x**2
            >>> y.d(y)
            0.0

            >>> y.trace_me()
            >>> y.d(y)
            1.0
        """
        if self not in self._lc:
            self._lc[self] = 1.0
            self._qc[self] = 0.0

    @property
    def real(self) -> float:
        """Return the real component of the underlying value.

        Returns
        -------
        float
            Real component of ``self.x``.
        """
        return self.x.real

    @property
    def imag(self) -> float:
        """Return the imaginary component of the underlying value.

        Returns
        -------
        float
            Imaginary component of ``self.x``.
        """
        return self.x.imag

    def _to_general_representation(self, str_func: object) -> str:
        """
        Provide the general representation of the underlying numeric object.

        Assumes ``self.tag`` is a string object.

        Returns
        -------
        str
            String representation produced via ``str_func``.
        """
        if self.tag is None:
            return f"ad({str_func(self.x)})"
        return f"ad({str_func(self.x)}, {self.tag!s})"

    def __repr__(self) -> str:
        return self._to_general_representation(repr)

    def __str__(self) -> str:
        return self._to_general_representation(str)

    def d(self, x: object = None) -> object:
        """
        Return first derivative with respect to ``x`` (an AD object).

        Parameters
        ----------
        x : AD object, optional
            ``x`` should be a single object created using the
            ``adnumber(...)`` constructor. If ``x=None``, then all associated
            first derivatives are returned in the form of a ``dict``.

        Returns
        -------
        df/dx : scalar or dict
            The derivative (if it exists), otherwise zero. When ``x`` is
            ``None``, returns the full derivative dictionary.
        """
        if x is not None:
            if isinstance(x, ADF):
                try:
                    tmp = self._lc[x]
                except KeyError:
                    tmp = 0.0
                return tmp if tmp.imag else tmp.real
            return 0.0
        return self._lc

    def d2(self, x: object = None) -> object:
        """
        Return pure second derivative with respect to ``x`` (an AD object).

        Parameters
        ----------
        x : AD object, optional
            ``x`` should be a single object created using the
            ``adnumber(...)`` constructor. If ``x=None``, then all associated
            second derivatives are returned in the form of a ``dict``.

        Returns
        -------
        d2f/dx2 : scalar or dict
            The pure second derivative (if it exists), otherwise zero. When
            ``x`` is ``None``, returns the full second-derivative dictionary.
        """
        if x is not None:
            if isinstance(x, ADF):
                try:
                    tmp = self._qc[x]
                except KeyError:
                    tmp = 0.0
                return tmp if tmp.imag else tmp.real
            return 0.0
        return self._qc

    def d2c(self, x: object = None, y: object = None) -> object:
        """
        Return cross-product second derivative with respect to ``x`` and ``y``.

        If both inputs are ``None``, then a dict containing all cross-product
        second derivatives is returned. This is one-way only (i.e., if
        ``f = f(x, y)`` then either ``d2f/dxdy`` or ``d2f/dydx`` will be in
        that dictionary and **not** both).

        If only one of the inputs is ``None`` or if the cross-product
        derivative doesn't exist, then zero is returned.

        If ``x`` and ``y`` are the same object, then the pure second-order
        derivative is returned.

        Parameters
        ----------
        x : AD object, optional
            First differentiation variable.
        y : AD object, optional
            Second differentiation variable.

        Returns
        -------
        d2f/dxdy : scalar or dict
            The cross-product derivative (if it exists), otherwise zero. When
            both inputs are ``None``, the full cross-product dictionary is
            returned.
        """
        if (x is not None) and (y is not None):
            if x is y:
                tmp = self.d2(x)
            elif isinstance(x, ADF) and isinstance(y, ADF):
                try:
                    tmp = self._cp[x, y]
                except KeyError:
                    try:
                        tmp = self._cp[y, x]
                    except KeyError:
                        tmp = 0.0
            else:
                tmp = 0.0

            return tmp if tmp.imag else tmp.real

        if ((x is not None) and not (y is not None)) or (
            (y is not None) and not (x is not None)
        ):
            return 0.0
        return self._cp

    def gradient(self, variables: object) -> list:
        """
        Return the gradient of the AD object given some input variables.

        The order of the inputs determines the order of the returned list of
        values::

            f.gradient([y, x, z]) --> [df/dy, df/dx, df/dz]

        Parameters
        ----------
        variables : array-like
            An array of objects (they don't have to be AD objects). If a
            partial derivative doesn't exist, then zero will be returned. If
            a single object is input, a single derivative will be returned as
            a list.

        Returns
        -------
        grad : list
            A list of partial derivatives.
        """
        try:
            grad = [self.d(v) for v in variables]
        except TypeError:
            grad = [self.d(variables)]
        return grad

    def hessian(self, variables: object) -> list:
        """
        Return the Hessian of the AD object given some input variables.

        The output order is determined by the input order::

            f.hessian([y, x, z]) --> [[d2f/dy2, d2f/dydx, d2f/dydz],
                                      [d2f/dxdy, d2f/dx2, d2f/dxdz],
                                      [d2f/dzdy, d2f/dzdx, d2f/dz2]]

        Parameters
        ----------
        variables : array-like
            An array of objects (they don't have to be AD objects).

        Returns
        -------
        hess : 2d-list
            A nested list of second partial derivatives (pure and
            cross-product).
        """
        try:
            hess = []
            for v1 in variables:
                hess.append([self.d2c(v1, v2) for v2 in variables])
        except TypeError:
            hess = [[self.d2(variables)]]
        return hess

    def sqrt(self) -> "ADF":
        """
        Return the square root of ``self``.

        A convenience function equal to ``self ** 0.5``. Required for some
        ``numpy`` functions like ``numpy.sqrt`` and ``numpy.std``.

        Returns
        -------
        ADF
            ``self ** 0.5``.
        """
        return self**0.5

    @staticmethod
    def _get_variables(ad_funcs: list) -> set:
        """Collect ADV objects involved in ``ad_funcs``.

        Parameters
        ----------
        ad_funcs : list
            List of ADF or ADV objects.

        Returns
        -------
        set
            Union of all variables referenced from each expression.
        """
        variables = set()
        for expr in ad_funcs:
            variables |= set(expr._lc)
        return variables

    def to_int(self) -> "ADF":
        """
        Convert the base number to an ``int`` object.

        Returns
        -------
        ADF
            ``self`` after coercing ``self.x`` to ``int``.
        """
        self.x = int(self.x)
        return self

    def to_float(self) -> "ADF":
        """
        Convert the base number to a ``float`` object.

        Returns
        -------
        ADF
            ``self`` after coercing ``self.x`` to ``float``.
        """
        self.x = float(self.x)
        return self

    def to_complex(self) -> "ADF":
        """
        Convert the base number to a ``complex`` object.

        Returns
        -------
        ADF
            ``self`` after coercing ``self.x`` to ``complex``.
        """
        self.x = complex(self.x)
        return self

    def __int__(self) -> int:
        return int(self.x)

    def __float__(self) -> float:
        return float(self.x)

    def __complex__(self) -> complex:
        return complex(self.x)

    def __bool__(self) -> bool:
        return bool(self.x)


class ADV(ADF):
    """
    A convenience class distinguishing FUNCTIONS (ADF) from VARIABLES.
    """

    def __init__(self, value: object, tag: str | None = None) -> None:
        super().__init__(value, {self: 1.0}, {self: 0.0}, {}, tag=tag)


def adnumber(x: object, tag: str | None = None) -> object:
    """
    Construct an automatic differentiation (AD) variable.

    Numbers that keep track of the derivatives of subsequent calculations.

    Parameters
    ----------
    x : scalar or array-like
        The nominal value(s) of the variable(s). Any numeric type or array is
        supported. If ``x`` is another AD object, a fresh copy is returned
        that contains all the derivatives of ``x``, but is not related to
        ``x`` in any way.
    tag : str, optional
        A string identifier. If an array of values for ``x`` is input, the
        tag applies to all the new AD objects.

    Returns
    -------
    x_ad : an AD object
        Newly constructed AD variable, or an array of them when ``x`` is
        array-like.
    """
    if numpy_installed and isinstance(x, numpy.ndarray):
        return numpy.array([adnumber(xi, tag) for xi in x])
    if isinstance(x, (tuple, list)):
        return type(x)([adnumber(xi, tag) for xi in x])

    if isinstance(x, ADF):
        return copy.deepcopy(x)
    if isinstance(x, CONSTANT_TYPES):
        return ADV(x, tag)

    raise NotImplementedError(
        f"Automatic differentiation not yet supported for {type(x)} objects"
    )


adfloat = adnumber


def gh(func: object) -> tuple[object, object]:
    """
    Generate gradient (g) and hessian (h) functions of an input function.

    Uses automatic differentiation. This is primarily for use in conjunction
    with the ``scipy.optimize`` package, though certainly not restricted
    there.

    NOTE: If NumPy is installed, the returned object from ``grad`` and
    ``hess`` will be a NumPy array. Otherwise, a generic list (or nested
    list, for ``hess``) will be returned.

    Parameters
    ----------
    func : callable
        This function should be composed of pure python mathematics (i.e.,
        it shouldn't be used for calling an external executable since AD
        doesn't work for that).

    Returns
    -------
    grad : callable
        The AD-compatible gradient function of ``func``.
    hess : callable
        The AD-compatible hessian function of ``func``.
    """

    def grad(x: object, *args: object) -> object:
        xa = adnumber(x)
        if numpy_installed and isinstance(x, numpy.ndarray):
            ans = func(xa, *args)
            if isinstance(ans, numpy.ndarray):
                return numpy.array(ans[0].gradient(list(xa)))
            return numpy.array(ans.gradient(list(xa)))
        try:
            return func(xa, *args).gradient(xa)
        except TypeError:
            return func(xa, *args).gradient([xa])

    def hess(x: object, *args: object) -> object:
        xa = adnumber(x)
        if numpy_installed and isinstance(x, numpy.ndarray):
            ans = func(xa, *args)
            if isinstance(ans, numpy.ndarray):
                return numpy.array(ans[0].hessian(list(xa)))
            return numpy.array(ans.hessian(list(xa)))
        try:
            return func(xa, *args).hessian(xa)
        except TypeError:
            return func(xa, *args).hessian([xa])

    for f, name in zip([grad, hess], ["gradient", "hessian"], strict=False):
        f.__doc__ = f"The {name} of {func.__name__}, "
        f.__doc__ += "calculated using automatic\ndifferentiation.\n\n"
        if func.__doc__ is not None and isinstance(func.__doc__, str):
            f.__doc__ += "Original documentation:\n" + func.__doc__

    return grad, hess


def jacobian(adfuns: object, advars: object) -> list:
    """
    Calculate the Jacobian matrix.

    Parameters
    ----------
    adfuns : array
        An array of AD objects (best when they are DEPENDENT AD variables).
    advars : array
        An array of AD objects (best when they are INDEPENDENT AD
        variables).

    Returns
    -------
    jac : 2d-array
        Each row is the gradient of each ``adfun`` with respect to each
        ``advar``, all in the order specified for both.
    """
    try:
        adfuns[0]
    except (TypeError, AttributeError):
        adfuns = [adfuns]

    try:
        advars[0]
    except (TypeError, AttributeError):
        advars = [advars]

    jac = []
    for adfun in adfuns:
        if hasattr(adfun, "gradient"):
            jac.append(adfun.gradient(advars))
        else:
            jac.append([0.0] * len(advars))

    return jac


if numpy_installed:

    def d(a: object, b: object, out: object = None) -> object:
        """
        Take a derivative of ``a`` with respect to ``b``.

        This is a numpy ufunc, so the derivative will be broadcast over both
        ``a`` and ``b``.

        Parameters
        ----------
        a : scalar or array
            Value(s) over which to take the derivative.
        b : scalar or array
            Variable(s) to take the derivative with respect to.
        out : array, optional
            Output array.

        Returns
        -------
        derivative : ndarray
            Derivative array broadcast over ``a`` and ``b``.
        """
        it = numpy.nditer(
            [a, b, out],
            flags=["buffered", "refs_ok"],
            op_flags=[
                ["readonly"],
                ["readonly"],
                ["writeonly", "allocate", "no_broadcast"],
            ],
        )
        for y, x, deriv in it:
            (v1,), (v2,) = y.flat, x.flat
            deriv[...] = v1.d(v2)
        return it.operands[2]

    def d2(a: object, b: object, out: object = None) -> object:
        """
        Take a second derivative of ``a`` with respect to ``b``.

        This is a numpy ufunc, so the derivative will be broadcast over both
        ``a`` and ``b``.

        Parameters
        ----------
        a : scalar or array
            Value(s) over which to take the second derivative.
        b : scalar or array
            Variable(s) to take the second derivative with respect to.
        out : array, optional
            Output array.

        Returns
        -------
        second_derivative : ndarray
            Second derivative array broadcast over ``a`` and ``b``.
        """
        it = numpy.nditer(
            [a, b, out],
            flags=["buffered", "refs_ok"],
            op_flags=[
                ["readonly"],
                ["readonly"],
                ["writeonly", "allocate", "no_broadcast"],
            ],
        )
        for y, x, deriv in it:
            (v1,), (v2,) = y.flat, x.flat
            deriv[...] = v1.d2(v2)
        return it.operands[2]
