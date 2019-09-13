from math import isinf

from gwt import LOG


def test_nlog():

    assert isinf(LOG(0.0))
    assert LOG(1.0) == 0.0
    assert abs(LOG(0.5) + 0.6931471805599453) < 1e-7
