from math import isinf

from gwt import nlog


def test_nlog():

    assert isinf(nlog(0.0))
    assert nlog(1.0) == 0.0
    assert abs(nlog(0.5) - 0.6931471805599453) < 1e-7
