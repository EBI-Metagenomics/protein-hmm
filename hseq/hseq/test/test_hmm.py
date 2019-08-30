from math import log, inf
from numpy.testing import assert_allclose
from numpy.random import RandomState
from hseq import SilentState, NormalState, TripletState, FrameState
from hseq import HMM


def test_hmm_a():
    alphabet = "ACGU"
    hmm = HMM(alphabet)
    start_state = SilentState("S", alphabet, False)
    hmm.add_state(start_state)

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state)

    hmm.normalize()

    assert_allclose(hmm.init_prob("S"), 0.5)
    assert_allclose(hmm.init_prob("E"), 0.5)
    assert_allclose(hmm.trans("S", "S"), 0.5)
    assert_allclose(hmm.trans("S", "E"), 0.5)
    assert_allclose(hmm.trans("E", "S"), 0.5)
    assert_allclose(hmm.trans("E", "E"), 0.5)


def test_hmm_b():
    alphabet = "ACGU"

    hmm = HMM(alphabet)
    start_state = SilentState("S", alphabet, False)
    hmm.add_state(start_state, inf)

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state, inf)

    hmm.set_trans("S", "E", -log(0.1))
    hmm.normalize()
    assert_allclose(hmm.trans("S", "E"), 1.0)
    assert_allclose(hmm.trans("S", "S"), 0.0)
    assert_allclose(hmm.init_prob("S"), 0.5)
    assert_allclose(hmm.init_prob("E"), 0.5)
    pass


def test_hmm_c():
    alphabet = "ACGU"

    hmm = HMM(alphabet)
    start_state = SilentState("S", alphabet, False)
    hmm.add_state(start_state, inf)

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state, 0.0)

    hmm.set_trans("S", "E", inf)
    hmm.normalize()

    assert_allclose(hmm.trans("S", "E"), 0.5)
    assert_allclose(hmm.trans("S", "S"), 0.5)
    assert_allclose(hmm.init_prob("S"), 0.0)
    assert_allclose(hmm.init_prob("E"), 1.0)


def test_hmm_d():
    alphabet = "ACGU"

    hmm = HMM(alphabet)
    start_state = SilentState("S", alphabet, False)
    hmm.add_state(start_state)

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state)

    hmm.set_trans("S", "E", 0.0)
    hmm.normalize()
    assert_allclose(hmm.trans("S", "E"), 1.0)
    assert_allclose(hmm.trans("S", "S"), 0.0)
