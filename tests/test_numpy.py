import pytest

import ad
from ad import adnumber
from ad.admath import exp, sin


np = pytest.importorskip("numpy")
numpy_testing = pytest.importorskip("numpy.testing")


def test_deriv_with_numpy_arrays() -> None:
    x = adnumber(2)
    x_row = adnumber(np.linspace(0, 2, 5))
    y = np.logspace(0, 4, 5)

    z = y * x
    dz = ad.d(z, x)
    numpy_testing.assert_allclose(np.array(dz, dtype=float), y)

    z = x_row**2
    dz = ad.d(z, x_row)
    numpy_testing.assert_allclose(
        np.array(dz, dtype=float), [0.0, 1.0, 2.0, 3.0, 4.0]
    )

    z = y * x_row
    dz = ad.d(z, x_row)
    numpy_testing.assert_allclose(np.array(dz, dtype=float), y)


def test_second_derivative_with_numpy_arrays() -> None:
    x = adnumber(2)
    x_row = adnumber(np.linspace(0, 2, 5))
    y = np.logspace(0, 4, 5)

    z = y * exp(-2 * x)
    dz = ad.d(z, x)
    ddz = ad.d2(z, x)
    numpy_testing.assert_allclose(
        np.array(dz, dtype=float), np.array(-2 * z, dtype=float)
    )
    numpy_testing.assert_allclose(
        np.array(ddz, dtype=float), np.array(4 * z, dtype=float)
    )

    z = x_row**2
    ddz = ad.d2(z, x_row)
    numpy_testing.assert_allclose(np.array(ddz, dtype=float), 2.0)

    z = y * sin(2 * x_row)
    ddz = ad.d2(z, x_row)
    numpy_testing.assert_allclose(
        np.array(ddz, dtype=float), np.array(-4 * z, dtype=float)
    )
