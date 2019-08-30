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


def test_hmm_emit_a():
    alphabet = "ACGU"

    hmm = HMM(alphabet)
    start_state = SilentState("S", alphabet, False)
    hmm.add_state(start_state, 0.0)

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state)

    M1 = NormalState("M1", {"A": -log(0.8), "C": -log(0.2), "G": inf, "U": inf})
    hmm.add_state(M1, inf)

    hmm.set_trans("S", "E", inf)
    hmm.set_trans("S", "M1", 0.0)
    hmm.set_trans("M1", "E", 0.0)
    hmm.normalize()

    random = RandomState(0)
    states, sequence = hmm.emit(random)
    assert states == "<S><M1><E>"
    assert sequence == "A"

    M2 = TripletState("M2", alphabet, {"AGU": -log(0.8), "AGG": -log(0.2)})
    hmm.add_state(M2, inf)

    hmm.set_trans("M1", "M2", 0.0)
    hmm.set_trans("M1", "E", inf)
    hmm.set_trans("M2", "E", 0.0)
    hmm.normalize()
    states, sequence = hmm.emit(random)
    assert states == "<S><M1><M2><E>"
    assert sequence == "AAGG"

    states, sequence = hmm.emit(random)
    assert states == "<S><M1><M2><E>"
    assert sequence == "AAGU"

    states, sequence = hmm.emit(random)
    assert states == "<S><M1><M2><E>"
    assert sequence == "AAGG"

    states, sequence = hmm.emit(random)
    assert states == "<S><M1><M2><E>"
    assert sequence == "AAGU"

    states, sequence = hmm.emit(random)
    assert states == "<S><M1><M2><E>"
    assert sequence == "AAGU"
