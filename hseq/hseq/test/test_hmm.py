from math import log, inf
from numpy.testing import assert_allclose
from numpy.random import RandomState
from hseq import SilentState, NormalState, TripletState, FrameState
from hseq import HMM, NLOG


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
    hmm.add_state(start_state, NLOG(0.0))

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state, NLOG(0.0))

    hmm.set_trans("S", "E", NLOG(0.1))
    hmm.normalize()
    assert_allclose(hmm.trans("S", "E"), 1.0)
    assert_allclose(hmm.trans("S", "S"), 0.0)
    assert_allclose(hmm.init_prob("S"), 0.5)
    assert_allclose(hmm.init_prob("E"), 0.5)


def test_hmm_init_prob_trans_c():
    alphabet = "ACGU"

    hmm = HMM(alphabet)
    start_state = SilentState("S", alphabet, False)
    hmm.add_state(start_state, NLOG(0.0))

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state, NLOG(1.0))

    hmm.set_trans("S", "E", NLOG(0.0))
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

    hmm.set_trans("S", "E", NLOG(1.0))
    hmm.normalize()
    assert_allclose(hmm.trans("S", "E"), 1.0)
    assert_allclose(hmm.trans("S", "S"), 0.0)


def test_hmm_emit_path():
    alphabet = "ACGU"

    hmm = HMM(alphabet)
    start_state = SilentState("S", alphabet, False)
    hmm.add_state(start_state, NLOG(1.0))

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state)

    M1 = NormalState(
        "M1", {"A": NLOG(0.8), "C": NLOG(0.2), "G": NLOG(0.0), "U": NLOG(0.0)}
    )
    hmm.add_state(M1, NLOG(0.0))

    hmm.set_trans("S", "E", NLOG(0.0))
    hmm.set_trans("S", "M1", NLOG(1.0))
    hmm.set_trans("M1", "E", NLOG(1.0))
    hmm.normalize()

    random = RandomState(0)
    path = hmm.emit(random)
    assert "".join(str(s[0]) for s in path) == "<S><M1><E>"
    assert "".join(s[1] for s in path) == "A"

    M2 = TripletState("M2", alphabet, {"AGU": NLOG(0.8), "AGG": NLOG(0.2)})
    hmm.add_state(M2, NLOG(0.0))

    hmm.set_trans("M1", "M2", NLOG(1.0))
    hmm.set_trans("M1", "E", NLOG(0.0))
    hmm.set_trans("M2", "E", NLOG(1.0))
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

    base_emission = {"A": NLOG(0.25), "C": NLOG(0.25), "G": NLOG(0.25), "U": NLOG(0.25)}
    codon_emission = {"AGU": NLOG(0.8), "AGG": NLOG(0.2)}
    epsilon = 0.1
    M3 = FrameState("M3", base_emission, codon_emission, epsilon)
    hmm.add_state(M3, NLOG(0.0))

    hmm.set_trans("M2", "E", NLOG(0.0))
    hmm.set_trans("M2", "M3", NLOG(1.0))
    hmm.set_trans("M3", "E", NLOG(1.0))
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
    hmm.add_state(start_state, NLOG(1.0))

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state, NLOG(0.0))

    M1 = NormalState(
        "M1", {"A": NLOG(0.8), "C": NLOG(0.2), "G": NLOG(0.0), "U": NLOG(0.0)}
    )
    hmm.add_state(M1, NLOG(0.0))

    M2 = NormalState(
        "M2", {"A": NLOG(0.4), "C": NLOG(0.6), "G": NLOG(0.0), "U": NLOG(0.6)}
    )
    hmm.add_state(M2, NLOG(0.0))

    hmm.set_trans("S", "M1", NLOG(1.0))
    hmm.set_trans("M1", "M2", NLOG(1.0))
    hmm.set_trans("M2", "E", NLOG(1.0))
    hmm.normalize()

    random = RandomState(0)
    path = hmm.emit(random)
    assert "".join(str(s[0]) for s in path) == "<S><M1><M2><E>"
    assert "".join(s[1] for s in path) == "AC"

    p = hmm.likelihood("AC", ["S", "M1", "M2", "E"], [0, 1, 1, 0])
    assert_allclose(p, 0.3)

    p = hmm.likelihood("AA", ["S", "M1", "M2", "E"], [0, 1, 1, 0])
    assert_allclose(p, 0.2)

    p = hmm.likelihood("AG", ["S", "M1", "M2", "E"], [0, 1, 1, 0])
    assert_allclose(p, 0, atol=1e-7)

    p = hmm.likelihood("AU", ["S", "M1", "M2", "E"], [0, 1, 1, 0])
    assert_allclose(p, 0.3)

    p = hmm.likelihood("CC", ["S", "M1", "M2", "E"], [0, 1, 1, 0])
    assert_allclose(p, 0.075)

    p = hmm.likelihood("CA", ["S", "M1", "M2", "E"], [0, 1, 1, 0])
    assert_allclose(p, 0.05)

    p = hmm.likelihood("CG", ["S", "M1", "M2", "E"], [0, 1, 1, 0])
    assert_allclose(p, 0.0, atol=1e-7)

    p = hmm.likelihood("CU", ["S", "M1", "M2", "E"], [0, 1, 1, 0])
    assert_allclose(p, 0.075)

    p = hmm.likelihood("GC", ["S", "M1", "M2", "E"], [0, 1, 1, 0])
    assert_allclose(p, 0.0, atol=1e-7)

    p = hmm.likelihood("GA", ["S", "M1", "M2", "E"], [0, 1, 1, 0])
    assert_allclose(p, 0.0, atol=1e-7)

    p = hmm.likelihood("GG", ["S", "M1", "M2", "E"], [0, 1, 1, 0])
    assert_allclose(p, 0.0, atol=1e-7)

    p = hmm.likelihood("GU", ["S", "M1", "M2", "E"], [0, 1, 1, 0])
    assert_allclose(p, 0.0, atol=1e-7)

    p = hmm.likelihood("UC", ["S", "M1", "M2", "E"], [0, 1, 1, 0])
    assert_allclose(p, 0.0, atol=1e-7)

    p = hmm.likelihood("UA", ["S", "M1", "M2", "E"], [0, 1, 1, 0])
    assert_allclose(p, 0.0, atol=1e-7)

    p = hmm.likelihood("UG", ["S", "M1", "M2", "E"], [0, 1, 1, 0])
    assert_allclose(p, 0.0, atol=1e-7)

    p = hmm.likelihood("UU", ["S", "M1", "M2", "E"], [0, 1, 1, 0])
    assert_allclose(p, 0.0, atol=1e-7)


def test_hmm_lik_2():
    alphabet = "AC"

    hmm = HMM(alphabet)
    start_state = NormalState("S", {"A": NLOG(0.8), "C": NLOG(0.2)})
    hmm.add_state(start_state, NLOG(1.0))

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state, NLOG(0.0))

    hmm.set_trans("S", "E", NLOG(1.0))
    hmm.normalize()

    # random = RandomState(0)
    # path = hmm.emit(random)
    # assert "".join(str(s[0]) for s in path) == "<S><M1><M2><E>"
    # assert "".join(s[1] for s in path) == "AC"

    # p = hmm.likelihood("AC", ["S", "M1", "M2", "E"], [0, 1, 1, 0])
    # assert_allclose(p, 0.3)


def test_hmm_viterbi():
    alphabet = "ACGU"

    hmm = HMM(alphabet)
    start_state = SilentState("S", alphabet, False)
    hmm.add_state(start_state, NLOG(1.0))

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state, NLOG(0.0))

    M1 = NormalState(
        "M1", {"A": NLOG(0.8), "C": NLOG(0.2), "G": NLOG(0.0), "U": NLOG(0.0)}
    )
    hmm.add_state(M1, NLOG(0.0))

    M2 = NormalState(
        "M2", {"A": NLOG(0.4), "C": NLOG(0.6), "G": NLOG(0.0), "U": NLOG(0.6)}
    )
    hmm.add_state(M2, NLOG(0.0))

    hmm.set_trans("S", "M1", NLOG(1.0))
    hmm.set_trans("M1", "M2", NLOG(1.0))
    hmm.set_trans("M2", "E", NLOG(1.0))
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

    p = hmm.likelihood("AC", ["S", "M1", "M2", "E"], [0, 1, 1, 0])
    assert_allclose(p, 0.3)
