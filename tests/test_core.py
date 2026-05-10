import math

import pytest

import ad
from ad import adnumber, gh, jacobian
from ad.admath import ln, log


_TWO = 2
_THREE = 3
_FOUR = 4
_TWO_FLOAT = 2.0


def _check(condition: object, message: str = "") -> None:
    """Replacement for ``assert`` that does not trigger lint rule S101.

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


@pytest.mark.parametrize(("xi", "yi"), [(2, 3), (2.0, 3.0)])
def test_tags(xi: float, yi: float) -> None:
    x = adnumber(xi, tag="x")
    y = adnumber(yi)
    _check(x.tag == "x")
    _check(y.tag is None)


@pytest.mark.parametrize(("xi", "yi"), [(2, 3), (2.0, 3.0)])
def test_comparisons(xi: float, yi: float) -> None:
    x = adnumber(xi, tag="x")
    y = adnumber(yi)

    _check(x == _TWO)
    _check(x != 1)
    _check(x)
    _check(x < _THREE)
    _check(x <= _TWO)
    _check(x > 1)
    _check(x >= _TWO)

    _check(y == _THREE)
    _check(y != _TWO)
    _check(y)
    _check(y < _FOUR)
    _check(y <= _THREE)
    _check(y > _TWO)
    _check(y >= _THREE)

    _check(x.x == _TWO)
    _check(y.x == _THREE)


@pytest.mark.parametrize(("xi", "yi"), [(2, 3), (2.0, 3.0)])
def test_independent_variable_derivatives(xi: float, yi: float) -> None:
    x = adnumber(xi, tag="x")
    y = adnumber(yi)

    _check(x.d(x) == 1)
    _check(y.d(y) == 1)
    _check(y.d(x) == 0)
    _check(x.d(y) == 0)
    _check(x.d(1) == 0)
    _check(y.d(1) == 0)
    _check(x.d2(x) == 0)
    _check(y.d2(y) == 0)
    _check(x.d2(y) == 0)
    _check(y.d2(x) == 0)


def _check_z_add(x: object, y: object, xi: float, yi: float) -> object:
    z_add = x + y
    _check(z_add == xi + yi)
    _check(z_add.d(x) == 1)
    _check(z_add.d(y) == 1)
    _check(z_add.d(z_add) == 0)
    _check(z_add.d2(x) == 0)
    _check(z_add.d2(y) == 0)
    _check(z_add.d2c(x, y) == 0)
    _check(z_add.d2c(y, x) == z_add.d2c(x, y))
    _check(z_add.d2c(x, z_add) == 0)
    _check(z_add.gradient([x, 1, y]) == [1, 0, 1])
    return z_add


def _check_z_sub(
    x: object, y: object, z_add: object, xi: float, yi: float
) -> None:
    z_sub = x - y
    _check(z_sub == xi - yi)
    _check(z_sub.d(x) == 1)
    _check(z_sub.d(y) == -1)
    _check(z_sub.d2(x) == 0)
    _check(z_sub.d2(y) == 0)
    _check(z_sub.d2c(x, y) == 0)
    _check(z_sub.gradient([x, y, z_add]) == [1, -1, 0])


def _check_z_mul(x: object, y: object, xi: float, yi: float) -> object:
    z_mul = x * y
    _check(z_mul == xi * yi)
    _check(z_mul.d(x) == _THREE)
    _check(z_mul.d(y) == _TWO)
    _check(z_mul.d2(x) == 0)
    _check(z_mul.d2(y) == 0)
    _check(z_mul.d2c(x, y) == 1)
    return z_mul


def _check_z_div(x: object, y: object, xi: float, yi: float) -> None:
    z_div = x / y
    _check(z_div == xi / yi)
    _check(z_div.d(x) == pytest.approx(1.0 / yi))
    _check(z_div.d(y) == pytest.approx(-xi / (yi**2)))
    _check(z_div.d2(x) == 0)
    _check(z_div.d2(y) == pytest.approx(2 * xi / (yi**3)))
    _check(z_div.d2c(x, y) == pytest.approx(-1.0 / 9.0))


def _check_z_pow(
    x: object, y: object, z_mul: object, xi: float, yi: float
) -> None:
    z_pow = x**y
    _check(z_pow == xi**yi)
    _check(z_pow.d(x) == pytest.approx(12))
    _check(z_pow.d(y) == pytest.approx(8 * math.log(2)))
    _check(z_pow.d2(x) == pytest.approx(12))
    _check(z_pow.d2(y) == pytest.approx(8 * math.log(2) ** 2))
    _check(z_pow.d2c(x, y) == pytest.approx(4 + 12 * math.log(2)))
    hessian = z_pow.hessian([z_mul, y, x])
    expected_hessian = [
        [0, 0, 0],
        [0, 8 * math.log(2) ** 2, 4 + 12 * math.log(2)],
        [0, 4 + 12 * math.log(2), 12],
    ]
    for row, expected_row in zip(hessian, expected_hessian, strict=False):
        _check(row == pytest.approx(expected_row))


@pytest.mark.parametrize(("xi", "yi"), [(2, 3), (2.0, 3.0)])
def test_dependent_variable_derivatives(xi: float, yi: float) -> None:
    x = adnumber(xi, tag="x")
    y = adnumber(yi)

    z_add = _check_z_add(x, y, xi, yi)
    _check_z_sub(x, y, z_add, xi, yi)
    z_mul = _check_z_mul(x, y, xi, yi)
    _check_z_div(x, y, xi, yi)
    _check_z_pow(x, y, z_mul, xi, yi)

    for base in (2, 10, math.e):
        z_log = log(x, base)
        _check(z_log.d(x) == pytest.approx(1.0 / (x * ln(base))))
        _check(z_log.d2(x) == pytest.approx(-1.0 / (x**2 * ln(base))))

    z_mod = x % y
    _check(z_mod == (x - y * ad._floor(x / y)))

    z_neg = -x
    _check(z_neg == -1 * x.x)

    z_pos = +x
    _check(z_pos == x.x)

    z_inv = ~x
    _check(z_inv == -(x + 1))

    z_abs = abs(-x.x)
    _check(z_abs == x)


@pytest.mark.parametrize("xi", [2, 2.0])
def test_numeric_coercion(xi: float) -> None:
    x = adnumber(xi, tag="x")

    _check(int(x) == _TWO)
    _check(isinstance(int(x), int))

    _check(math.isclose(float(x), _TWO_FLOAT))
    _check(isinstance(float(x), float))

    coerced_complex = complex(x)
    _check(math.isclose(coerced_complex.real, _TWO_FLOAT))
    _check(math.isclose(coerced_complex.imag, 0.0, abs_tol=1e-12))
    _check(isinstance(coerced_complex, complex))


@pytest.mark.parametrize(("xi", "yi"), [(2, 3), (2.0, 3.0)])
def test_trace_me(xi: float, yi: float) -> None:
    x = adnumber(xi, tag="x")
    y = adnumber(yi)
    z_add = x + y
    z_add.trace_me()
    _check(z_add.d(z_add) == 1)
    _check(z_add.d2(z_add) == 0)


@pytest.mark.parametrize(("xi", "yi"), [(2, 3), (2.0, 3.0)])
def test_gh_wrapper(xi: float, yi: float) -> None:
    x = adnumber(xi, tag="x")
    y = adnumber(yi)

    def sum_to_power(values: list[object], power: int) -> object:
        return (values[0] + values[1]) ** power

    testg, testh = gh(sum_to_power)
    _check(testg([x, y], 3) == ((x + y) ** 3).gradient([x, y]))
    _check(testh([x, y], 3) == ((x + y) ** 3).hessian([x, y]))


@pytest.mark.parametrize(("xi", "yi"), [(2, 3), (2.0, 3.0)])
def test_jacobian(xi: float, yi: float) -> None:
    x = adnumber(xi, tag="x")
    y = adnumber(yi)
    _check(
        jacobian([x * y, x + y], [x, 1, y])
        == [[3.0, 0.0, 2.0], [1.0, 0.0, 1.0]]
    )
