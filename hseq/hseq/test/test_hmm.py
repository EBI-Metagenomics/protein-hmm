from math import log, inf
from numpy.testing import assert_allclose
from numpy.random import RandomState
from hseq import SilentState, NormalState, TripletState, FrameState
from hseq import HMM


def test_hmm_init_prob_trans_a():
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
    assert_allclose(hmm.trans("E", "S"), 0.0)
    assert_allclose(hmm.trans("E", "E"), 1.0)


def test_hmm_init_prob_trans_b():
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


def test_hmm_init_prob_trans_c():
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


def test_hmm_init_prob_trans_d():
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


def test_hmm_emit_path():
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
    path = hmm.emit(random)
    assert "".join(str(s[0]) for s in path) == "<S><M1><E>"
    assert "".join(s[1] for s in path) == "A"

    M2 = TripletState("M2", alphabet, {"AGU": -log(0.8), "AGG": -log(0.2)})
    hmm.add_state(M2, inf)

    hmm.set_trans("M1", "M2", 0.0)
    hmm.set_trans("M1", "E", inf)
    hmm.set_trans("M2", "E", 0.0)
    hmm.normalize()

    path = hmm.emit(random)
    assert "".join(str(s[0]) for s in path) == "<S><M1><M2><E>"
    assert "".join(s[1] for s in path) == "AAGG"

    path = hmm.emit(random)
    assert "".join(str(s[0]) for s in path) == "<S><M1><M2><E>"
    assert "".join(s[1] for s in path) == "AAGU"

    path = hmm.emit(random)
    assert "".join(str(s[0]) for s in path) == "<S><M1><M2><E>"
    assert "".join(s[1] for s in path) == "AAGG"

    path = hmm.emit(random)
    assert "".join(str(s[0]) for s in path) == "<S><M1><M2><E>"
    assert "".join(s[1] for s in path) == "AAGU"

    path = hmm.emit(random)
    assert "".join(str(s[0]) for s in path) == "<S><M1><M2><E>"
    assert "".join(s[1] for s in path) == "AAGU"

    base_emission = {"A": -log(0.25), "C": -log(0.25), "G": -log(0.25), "U": -log(0.25)}
    codon_emission = {"AGU": -log(0.8), "AGG": -log(0.2)}
    epsilon = 0.1
    M3 = FrameState("M3", base_emission, codon_emission, epsilon)
    hmm.add_state(M3, inf)

    hmm.set_trans("M2", "E", inf)
    hmm.set_trans("M2", "M3", 0.0)
    hmm.set_trans("M3", "E", 0.0)
    hmm.normalize()
    assert_allclose(hmm.trans("S", "M1"), 1)
    assert_allclose(hmm.trans("M1", "M2"), 1)
    assert_allclose(hmm.trans("M2", "M3"), 1)
    assert_allclose(hmm.trans("M3", "E"), 1)
    path = hmm.emit(random)
    assert "".join(str(s[0]) for s in path) == "<S><M1><M2><M3><E>"
    assert "".join(s[1] for s in path) == "AAGGAGU"


# def test_hmm_viterbi():
#     alphabet = "ACGU"

#     hmm = HMM(alphabet)
#     start_state = SilentState("S", alphabet, False)
#     hmm.add_state(start_state, 0.0)

#     end_state = SilentState("E", alphabet, True)
#     hmm.add_state(end_state, inf)

#     M1 = NormalState("M1", {"A": -log(0.8), "C": -log(0.2), "G": inf, "U": inf})
#     hmm.add_state(M1, inf)

#     M2 = NormalState("M2", {"A": -log(0.4), "C": -log(0.6), "G": inf, "U": -log(0.6)})
#     hmm.add_state(M2, inf)

#     hmm.set_trans("S", "M1", 0.0)
#     hmm.set_trans("M1", "M2", 0.0)
#     hmm.set_trans("M2", "E", 0.0)
#     hmm.normalize()

#     random = RandomState(0)
#     path = hmm.emit(random)
#     assert "".join(str(s[0]) for s in path) == "<S><M1><M2><E>"
#     assert "".join(s[1] for s in path) == "AC"

#     hmm.viterbi("AC")
