from math import log, inf
from numpy.testing import assert_allclose
from numpy.random import RandomState
from hseq import SilentState, NormalState, TripletState, FrameState
from hseq import HMM, nlog


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
    hmm.add_state(start_state, nlog(0.0))

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state, nlog(0.0))

    hmm.set_trans("S", "E", nlog(0.1))
    hmm.normalize()
    assert_allclose(hmm.trans("S", "E"), 1.0)
    assert_allclose(hmm.trans("S", "S"), 0.0)
    assert_allclose(hmm.init_prob("S"), 0.5)
    assert_allclose(hmm.init_prob("E"), 0.5)


def test_hmm_init_prob_trans_c():
    alphabet = "ACGU"

    hmm = HMM(alphabet)
    start_state = SilentState("S", alphabet, False)
    hmm.add_state(start_state, nlog(0.0))

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state, nlog(1.0))

    hmm.set_trans("S", "E", nlog(0.0))
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

    hmm.set_trans("S", "E", nlog(1.0))
    hmm.normalize()
    assert_allclose(hmm.trans("S", "E"), 1.0)
    assert_allclose(hmm.trans("S", "S"), 0.0)


def test_hmm_emit_path():
    alphabet = "ACGU"

    hmm = HMM(alphabet)
    start_state = SilentState("S", alphabet, False)
    hmm.add_state(start_state, nlog(1.0))

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state)

    M1 = NormalState(
        "M1", {"A": nlog(0.8), "C": nlog(0.2), "G": nlog(0.0), "U": nlog(0.0)}
    )
    hmm.add_state(M1, nlog(0.0))

    hmm.set_trans("S", "E", nlog(0.0))
    hmm.set_trans("S", "M1", nlog(1.0))
    hmm.set_trans("M1", "E", nlog(1.0))
    hmm.normalize()

    random = RandomState(0)
    path = hmm.emit(random)
    assert "".join(str(s[0]) for s in path) == "<S><M1><E>"
    assert "".join(s[1] for s in path) == "A"

    M2 = TripletState("M2", alphabet, {"AGU": nlog(0.8), "AGG": nlog(0.2)})
    hmm.add_state(M2, nlog(0.0))

    hmm.set_trans("M1", "M2", nlog(1.0))
    hmm.set_trans("M1", "E", nlog(0.0))
    hmm.set_trans("M2", "E", nlog(1.0))
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

    base_emission = {"A": nlog(0.25), "C": nlog(0.25), "G": nlog(0.25), "U": nlog(0.25)}
    codon_emission = {"AGU": nlog(0.8), "AGG": nlog(0.2)}
    epsilon = 0.1
    M3 = FrameState("M3", base_emission, codon_emission, epsilon)
    hmm.add_state(M3, nlog(0.0))

    hmm.set_trans("M2", "E", nlog(0.0))
    hmm.set_trans("M2", "M3", nlog(1.0))
    hmm.set_trans("M3", "E", nlog(1.0))
    hmm.normalize()
    assert_allclose(hmm.trans("S", "M1"), 1)
    assert_allclose(hmm.trans("M1", "M2"), 1)
    assert_allclose(hmm.trans("M2", "M3"), 1)
    assert_allclose(hmm.trans("M3", "E"), 1)
    path = hmm.emit(random)
    assert "".join(str(s[0]) for s in path) == "<S><M1><M2><M3><E>"
    assert "".join(s[1] for s in path) == "AAGGAGU"


def test_hmm_lik_1():
    alphabet = "ACGU"

    hmm = HMM(alphabet)
    start_state = SilentState("S", alphabet, False)
    hmm.add_state(start_state, nlog(1.0))

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state, nlog(0.0))

    M1 = NormalState(
        "M1", {"A": nlog(0.8), "C": nlog(0.2), "G": nlog(0.0), "U": nlog(0.0)}
    )
    hmm.add_state(M1, nlog(0.0))

    M2 = NormalState(
        "M2", {"A": nlog(0.4), "C": nlog(0.6), "G": nlog(0.0), "U": nlog(0.6)}
    )
    hmm.add_state(M2, nlog(0.0))

    hmm.set_trans("S", "M1", nlog(1.0))
    hmm.set_trans("M1", "M2", nlog(1.0))
    hmm.set_trans("M2", "E", nlog(1.0))
    hmm.normalize()

    random = RandomState(0)
    path = hmm.emit(random)
    assert "".join(str(s[0]) for s in path) == "<S><M1><M2><E>"
    assert "".join(s[1] for s in path) == "AC"

    p = hmm.likelihood("AC", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.3)

    p = hmm.likelihood("AA", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.2)

    p = hmm.likelihood("AG", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0, atol=1e-7)

    p = hmm.likelihood("AU", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.3)

    p = hmm.likelihood("CC", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.075)

    p = hmm.likelihood("CA", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.05)

    p = hmm.likelihood("CG", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.0, atol=1e-7)

    p = hmm.likelihood("CU", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.075)

    p = hmm.likelihood("GC", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.0, atol=1e-7)

    p = hmm.likelihood("GA", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.0, atol=1e-7)

    p = hmm.likelihood("GG", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.0, atol=1e-7)

    p = hmm.likelihood("GU", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.0, atol=1e-7)

    p = hmm.likelihood("UC", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.0, atol=1e-7)

    p = hmm.likelihood("UA", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.0, atol=1e-7)

    p = hmm.likelihood("UG", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.0, atol=1e-7)

    p = hmm.likelihood("UU", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.0, atol=1e-7)


def test_hmm_lik_2():
    alphabet = "AC"

    hmm = HMM(alphabet)
    start_state = NormalState("S", {"A": nlog(0.8), "C": nlog(0.2)})
    hmm.add_state(start_state, nlog(1.0))

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state, nlog(0.0))

    hmm.set_trans("S", "E", nlog(1.0))
    hmm.normalize()

    p = hmm.likelihood("A", [("S", 1), ("E", 0)])
    assert_allclose(p, 0.8)

    p = hmm.likelihood("C", [("S", 1), ("E", 0)])
    assert_allclose(p, 0.2)

    p = hmm.likelihood("A", [("S", 1)])
    assert_allclose(p, 0.8)

    p = hmm.likelihood("C", [("S", 1)])
    assert_allclose(p, 0.2)

    p = hmm.likelihood("C", [("S", 2)])
    assert_allclose(p, 0.0, atol=1e-7)

    p = hmm.likelihood("C", [("S", 0)])
    assert_allclose(p, 0.0, atol=1e-7)

    p = hmm.likelihood("C", [("E", 1)])
    assert_allclose(p, 0.0, atol=1e-7)

    p = hmm.likelihood("", [])
    assert_allclose(p, 1.0)

    p = hmm.likelihood("A", [])
    assert_allclose(p, 0.0)


def test_hmm_viterbi():
    alphabet = "ACGU"

    hmm = HMM(alphabet)
    start_state = SilentState("S", alphabet, False)
    hmm.add_state(start_state, nlog(1.0))

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state, nlog(0.0))

    M1 = NormalState(
        "M1", {"A": nlog(0.8), "C": nlog(0.2), "G": nlog(0.0), "U": nlog(0.0)}
    )
    hmm.add_state(M1, nlog(0.0))

    M2 = NormalState(
        "M2", {"A": nlog(0.4), "C": nlog(0.6), "G": nlog(0.0), "U": nlog(0.6)}
    )
    hmm.add_state(M2, nlog(0.0))

    hmm.set_trans("S", "M1", nlog(1.0))
    hmm.set_trans("M1", "M2", nlog(1.0))
    hmm.set_trans("M2", "E", nlog(1.0))
    hmm.normalize()

    random = RandomState(0)
    path = hmm.emit(random)
    assert "".join(str(s[0]) for s in path) == "<S><M1><M2><E>"
    assert "".join(s[1] for s in path) == "AC"

    lik, path = hmm.viterbi("AC")
    assert_allclose(lik, 0.3)
    assert "".join(str(s[0]) for s in path) == "<S><M1><M2><E>"
    assert path[0][1] == 0
    assert path[1][1] == 1
    assert path[2][1] == 1
    assert path[3][1] == 0

    p = hmm.likelihood("AC", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.3)
