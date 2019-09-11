from gwt._testit import test


def test_testit():
    assert test(verbose=False) == 0
