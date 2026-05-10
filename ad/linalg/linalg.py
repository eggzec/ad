"""
This sub-module allows for the usage of several linear algebra routines that
are otherwise unavailable for use for automatic differentiation. Not all
numpy.linalg routines have an equivalent here, but we're working on it.

Decompositions
    a. chol (Cholesky)
    b. qr (QR Factorization
    c. lu (LU Decomposition)

Solving equations and inverting matrices
    a. solve (Solve a linear matrix equation, or system of linear scalar eqs)
    b. lstsq (Solve a linear least squares problem)
    c. inv (Compute the (multiplicative) inverse of a matrix)

(c) 2013 by Abraham Lee <tisimst@gmail.com>.
Please send feature requests, bug reports, or feedback to this address.

This software is released under a dual license.  (1) The BSD license.
(2) Any other license, as long as it is obtained from the original
author.

"""

import numpy as np
from numpy import ndarray


__all__ = ["chol", "inv", "lstsq", "lu", "qr", "solve"]


def _ensure(condition: object, message: str = "") -> None:
    """Raise ``AssertionError`` when ``condition`` is falsy.

    Parameters
    ----------
    condition : object
        Truthy value to test.
    message : str
        Optional message reported when the condition is falsy.

    Raises
    ------
    AssertionError
        If ``condition`` is falsy.
    """
    if not condition:
        raise AssertionError(message)


def chol(a: ndarray, side: str = "lower") -> ndarray:
    """
    Cholesky decomposition of a symmetric, positive-definite matrix.

    The decomposition is defined by ``A = L @ L.T = U.T @ U``, where ``L`` is
    a lower triangular matrix and ``U`` is an upper triangular matrix.

    Parameters
    ----------
    a : 2d-array
        The input matrix.
    side : str
        If ``'lower'`` (default), then the lower triangular form of the
        decomposition is returned. If ``'upper'``, then the upper triangular
        form of the decomposition is returned (the transpose of ``'lower'``).

    Returns
    -------
    out : 2d-array
        The lower (or upper) triangular matrix that helps define the
        decomposition.
    """
    a = np.array(a)
    _ensure(a.shape[0] == a.shape[1], "Input matrix must be square")

    n = len(a)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(lower[i][k] * lower[j][k] for k in range(j))
            lower[i][j] = (
                (a[i][i] - s) ** 0.5
                if i == j
                else (1.0 / lower[j][j] * (a[i][j] - s))
            )

    if side == "lower":
        return np.array(lower)
    return np.array(lower).T


def qr(a: ndarray) -> tuple[ndarray, ndarray]:
    """
    QR Decomposition.

    Parameters
    ----------
    a : 2d-array
        The input matrix (need not be square).

    Returns
    -------
    qm : 2d-array
        The orthogonal Q matrix from the decomposition.
    rm : 2d-array
        The R matrix from the decomposition.
    """
    a = np.atleast_2d(a)
    m, n = a.shape
    qm = np.eye(m)
    rm = a.copy()
    for i in range(n - 1 if m == n else n):
        x = _get_submatrix(rm, i, i, m, i)
        h = np.eye(m)
        h = _set_submatrix(h, i, i, _householder(x))
        qm = np.dot(qm, h)
        rm = np.dot(h, rm)
    return qm, rm


def lu(a: ndarray) -> tuple[ndarray, ndarray, ndarray]:
    """
    Decompose an ``n x n`` matrix ``a`` by ``PA = LU``.

    Parameters
    ----------
    a : 2d-array
        The input matrix.

    Returns
    -------
    lower : 2d-array
        The lower-triangular matrix of the decomposition.
    upper : 2d-array
        The upper-triangular matrix of the decomposition.
    pivots : 2d-array
        The pivoting matrix used in the decomposition.
    """
    a = np.array(a)
    _ensure(a.shape[0] == a.shape[1], "Input matrix must be square")

    n = len(a)
    lower = [[0.0] * n for _ in range(n)]
    upper = [[0.0] * n for _ in range(n)]

    pivots = [[float(i == j) for i in range(n)] for j in range(n)]
    for j in range(n):
        row = max(range(j, n), key=lambda i, jj=j: a[i][jj])
        if j != row:
            pivots[j], pivots[row] = pivots[row], pivots[j]

    a2 = np.dot(pivots, a)
    for j in range(n):
        lower[j][j] = 1.0
        for i in range(j + 1):
            s1 = sum(upper[k][j] * lower[i][k] for k in range(i))
            upper[i][j] = a2[i][j] - s1
        for i in range(j, n):
            s2 = sum(upper[k][j] * lower[i][k] for k in range(j))
            lower[i][j] = (a2[i][j] - s2) / upper[j][j]
    return (np.array(lower), np.array(upper), np.array(pivots))


def _normalize_solve_inputs(
    a: ndarray, b: ndarray
) -> tuple[ndarray, ndarray, int]:
    """Validate and reshape inputs for :func:`solve`.

    Parameters
    ----------
    a : ndarray
        Coefficient matrix.
    b : ndarray
        Right-hand-side array.

    Returns
    -------
    matrix : ndarray
        Coefficient matrix as a ``np.matrix``.
    rhs : ndarray
        Right-hand-side array.
    numout : int
        Number of solution columns.

    Raises
    ------
    TypeError
        If ``a`` cannot be converted to a 2-dimensional array.
    """
    try:
        matrix = np.matrix(a)
    except (TypeError, ValueError) as err:
        raise TypeError("A must be a 2-dimensional array") from err
    _ensure(
        matrix.shape[0] >= matrix.shape[1],
        "A must not have less rows than columns",
    )
    rhs = np.array(b)
    numout = 1 if rhs.ndim == 1 else rhs.shape[1]
    _ensure(
        matrix.shape[0] == rhs.shape[0],
        "b must have the same number of rows as A",
    )
    return matrix, rhs, numout


def _forward_substitute(eqs: ndarray, n: int, m: int) -> None:
    """Apply forward substitution in place on the augmented matrix.

    Parameters
    ----------
    eqs : ndarray
        Augmented matrix that combines the coefficient matrix and the
        right-hand-side. Modified in place.
    n : int
        Number of rows.
    m : int
        Number of columns.
    """
    for k in range(n - 1):
        p = k
        piv_el = abs(eqs[k, k])

        for i in range(k + 1, n):
            tmp = abs(eqs[i, k])
            if tmp > piv_el:
                piv_el = tmp
                p = i

        if p != k:
            rtmp = np.array(eqs[p, :])
            eqs[p, :] = eqs[k, :]
            eqs[k, :] = rtmp

        for i in range(k + 1, n):
            f = 1.0 * eqs[i, k] / eqs[k, k]
            for j in range(k, m):
                eqs[i, j] -= f * eqs[k, j]


def _backward_substitute(eqs: ndarray, n: int, m: int) -> None:
    """Apply backward substitution in place on the augmented matrix.

    Parameters
    ----------
    eqs : ndarray
        Augmented matrix that combines the coefficient matrix and the
        right-hand-side. Modified in place.
    n : int
        Number of rows.
    m : int
        Number of columns.
    """
    for k in range(1, n)[::-1]:
        for i in range(k)[::-1]:
            f = 1.0 * eqs[i, k] / eqs[k, k]
            for j in range(m)[::-1]:
                eqs[i, j] -= f * eqs[k, j]

    for i in range(n):
        x = 1.0 * eqs[i, i]
        for j in range(m):
            eqs[i, j] /= x


def solve(a: ndarray, b: ndarray) -> ndarray:
    """
    Solve a system of equations ``Ax = b`` by Gaussian elimination.

    Parameters
    ----------
    a : 2d-array
        The LHS of the system of equations. The number of rows must not be
        less than the number of columns, but can be more.
    b : array-like
        The RHS of the system of equations (must have the same number of rows
        as ``a``). If more than one column is given, a solution is generated
        for each column.

    Returns
    -------
    x : array-like
        The solution that satisfies the equality ``a @ x == b``. If multiple
        ``b`` columns are given, a solution column will be returned for each
        set.
    """
    matrix, rhs, numout = _normalize_solve_inputs(a, b)

    if matrix.shape[0] > matrix.shape[1]:
        rhs = matrix.T * rhs
        matrix = matrix.T * matrix

    eqs = np.column_stack((matrix, rhs))
    n, m = eqs.shape

    _forward_substitute(eqs, n, m)
    _backward_substitute(eqs, n, m)

    if numout == 1:
        return np.array(eqs[:, -(m - n) :]).ravel()
    return np.array(eqs[:, -(m - n) :])


def lstsq(a: ndarray, b: ndarray) -> ndarray:
    """
    Solve ``Ax = b`` with the linear least squares method.

    Parameters
    ----------
    a : 2d-array
        The linear system matrix.
    b : array
        The right-hand-side of the equation.

    Returns
    -------
    x : array
        The solution to the system of equations.
    """
    q, r = qr(a)
    n = r.shape[1]
    x = _solve_upper_triangular(
        _get_submatrix(r, 0, 0, n - 1, n - 1), np.dot(q.T, b)
    )
    return x.ravel()


def inv(a: ndarray) -> ndarray:
    """
    Calculate the multiplicative inverse of a matrix.

    Parameters
    ----------
    a : 2d-array
        The input matrix.

    Returns
    -------
    inverse : 2d-array
        The inverse of ``a``.
    """
    shape = np.atleast_2d(a).shape
    _ensure(shape[0] == shape[1], "Matrix must be square")
    return solve(a, np.eye(shape[0]))


def _ematrix(m: int, n: int, x: float, i: int, j: int) -> ndarray:
    a = np.empty((m, n))
    a[:, :] = 0.0
    a[i, j] = x
    return a


def _unit_vector(n: int) -> ndarray:
    return _ematrix(n, 1, 1, 0, 0)


def _sign_value(r: float) -> int:
    if r < 0:
        return -1
    if r == 0:
        return 0
    return 1


def _householder(a: ndarray) -> ndarray:
    m = len(a)
    u = a - np.sqrt(np.dot(a.T, a)[0, 0]) * _unit_vector(m)
    v = u
    beta = 2 / np.dot(v.T, v)
    return np.eye(m) - beta * (np.dot(v, v.T).T)


def _get_submatrix(obj: ndarray, i1: int, j1: int, i2: int, j2: int) -> ndarray:
    """
    Return the submatrix bounded by inclusive index pairs.

    Parameters
    ----------
    obj : ndarray
        Source matrix.
    i1 : int
        Top row index (inclusive).
    j1 : int
        Left column index (inclusive).
    i2 : int
        Bottom row index (inclusive).
    j2 : int
        Right column index (inclusive).

    Returns
    -------
    submatrix : ndarray
        Submatrix view bounded by ``[i1, i2]`` rows and ``[j1, j2]`` columns.
    """
    return obj[i1 : i2 + 1, j1 : j2 + 1]


def _set_submatrix(obj: ndarray, i1: int, j1: int, subobj: ndarray) -> ndarray:
    """
    Insert a submatrix into ``obj`` starting at ``(i1, j1)``.

    Parameters
    ----------
    obj : ndarray
        Destination matrix; modified in place.
    i1 : int
        Top row index where insertion begins.
    j1 : int
        Left column index where insertion begins.
    subobj : ndarray
        Submatrix to insert.

    Returns
    -------
    obj : ndarray
        The modified destination matrix.
    """
    m, n = np.atleast_2d(subobj).shape
    obj[i1 : i1 + m, j1 : j1 + n] = subobj
    return obj


def _solve_upper_triangular(r: ndarray, b: ndarray) -> ndarray:
    r = np.atleast_2d(r)
    n = r.shape[1]

    x = np.zeros((n, 1))
    for k in range(n - 1, -1, -1):
        idx = min(n - 1, k)
        x[k, 0] = (
            b[k]
            - np.dot(
                _get_submatrix(r, k, idx, k, n - 1),
                _get_submatrix(x, idx, 0, n - 1, 0),
            )
        ) / r[k, k]
    return x


def _polyfit(x: ndarray, y: ndarray, n: int) -> ndarray:
    """
    Fit a polynomial of order ``n`` to data arrays ``x`` and ``y``.

    Parameters
    ----------
    x : array
        A data array that defines the first dimension coordinates.
    y : array
        A data array that defines the second dimension coordinates.
    n : int
        The order of the polynomial (e.g., 1 for linear, 2 for quadratic).

    Returns
    -------
    b : array
        The polynomial coefficients, in ascending order.
    """
    a = np.empty((len(x), n + 1))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a[i, j] = 1 if j == 0 else x[i] ** j
    return lstsq(a, y)
