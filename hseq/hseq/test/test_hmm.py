from math import isinf
import pytest
from numpy.testing import assert_allclose
from numpy.random import RandomState
from hseq import SilentState, NormalState, TripletState, FrameState
from hseq import HMM, LOG


def test_hmm_states():
    alphabet = "ACGU"

    hmm = HMM(alphabet)
    hmm.add_state(SilentState("S", alphabet, False))
    state = TripletState("M2", alphabet, {"AGU": LOG(0.8), "AGG": LOG(0.2)})
    hmm.add_state(state, LOG(0.0))

    states = hmm.states
    assert "S" in states
    assert "M2" in states
    assert len(states) == 2

    with pytest.raises(ValueError):
        hmm.add_state(NormalState("S", {a: LOG(1.0) for a in alphabet}))

    with pytest.raises(ValueError):
        hmm.add_state(NormalState("S2", {a: LOG(1.0) for a in alphabet[:-1]}))


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
    hmm.add_state(start_state, LOG(0.0))

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state, LOG(0.0))

    hmm.set_trans("S", "E", LOG(0.1))
    hmm.normalize()
    assert_allclose(hmm.trans("S", "E"), 1.0)
    assert_allclose(hmm.trans("S", "S"), 0.0)
    assert_allclose(hmm.init_prob("S"), 0.5)
    assert_allclose(hmm.init_prob("E"), 0.5)


def test_hmm_init_prob_trans_c():
    alphabet = "ACGU"

    hmm = HMM(alphabet)
    start_state = SilentState("S", alphabet, False)
    hmm.add_state(start_state, LOG(0.0))

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state, LOG(1.0))

    hmm.set_trans("S", "E", LOG(0.0))
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

    hmm.set_trans("S", "E", LOG(1.0))
    hmm.normalize()
    assert_allclose(hmm.trans("S", "E"), 1.0)
    assert_allclose(hmm.trans("S", "S"), 0.0)


def test_hmm_emit_path():
    alphabet = "ACGU"

    hmm = HMM(alphabet)
    start_state = SilentState("S", alphabet, False)
    hmm.add_state(start_state, LOG(1.0))

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state)

    M1 = NormalState("M1", {"A": LOG(0.8), "C": LOG(0.2), "G": LOG(0.0), "U": LOG(0.0)})
    hmm.add_state(M1, LOG(0.0))

    hmm.set_trans("S", "E", LOG(0.0))
    hmm.set_trans("S", "M1", LOG(1.0))
    hmm.set_trans("M1", "E", LOG(1.0))
    hmm.normalize()

    random = RandomState(0)
    path = hmm.emit(random)
    assert [p[0] for p in path] == ["S", "M1", "E"]
    assert "".join(s[1] for s in path) == "A"

    M2 = TripletState("M2", alphabet, {"AGU": LOG(0.8), "AGG": LOG(0.2)})
    hmm.add_state(M2, LOG(0.0))

    hmm.set_trans("M1", "M2", LOG(1.0))
    hmm.set_trans("M1", "E", LOG(0.0))
    hmm.set_trans("M2", "E", LOG(1.0))
    hmm.normalize()

    path = hmm.emit(random)
    assert [p[0] for p in path] == ["S", "M1", "M2", "E"]
    assert "".join(s[1] for s in path) == "AAGG"

    path = hmm.emit(random)
    assert [p[0] for p in path] == ["S", "M1", "M2", "E"]
    assert "".join(s[1] for s in path) == "AAGU"

    path = hmm.emit(random)
    assert [p[0] for p in path] == ["S", "M1", "M2", "E"]
    assert "".join(s[1] for s in path) == "AAGG"

    path = hmm.emit(random)
    assert [p[0] for p in path] == ["S", "M1", "M2", "E"]
    assert "".join(s[1] for s in path) == "AAGU"

    path = hmm.emit(random)
    assert [p[0] for p in path] == ["S", "M1", "M2", "E"]
    assert "".join(s[1] for s in path) == "AAGU"

    base_emission = {"A": LOG(0.25), "C": LOG(0.25), "G": LOG(0.25), "U": LOG(0.25)}
    codon_emission = {"AGU": LOG(0.8), "AGG": LOG(0.2)}
    epsilon = 0.1
    M3 = FrameState("M3", base_emission, codon_emission, epsilon)
    hmm.add_state(M3, LOG(0.0))

    hmm.set_trans("M2", "E", LOG(0.0))
    hmm.set_trans("M2", "M3", LOG(1.0))
    hmm.set_trans("M3", "E", LOG(1.0))
    hmm.normalize()
    assert_allclose(hmm.trans("S", "M1"), 1)
    assert_allclose(hmm.trans("M1", "M2"), 1)
    assert_allclose(hmm.trans("M2", "M3"), 1)
    assert_allclose(hmm.trans("M3", "E"), 1)
    path = hmm.emit(random)
    assert [p[0] for p in path] == ["S", "M1", "M2", "M3", "E"]
    assert "".join(s[1] for s in path) == "AAGGAGU"
    states_path = [("S", 0), ("M1", 1), ("M2", 3), ("M3", 3), ("E", 0)]
    loglik = hmm.likelihood("AAGGAGU", states_path, True)
    assert abs(loglik + 2.472373518623133) < 1e-7


def test_hmm_lik_1():
    alphabet = "ACGU"

    hmm = HMM(alphabet)
    start_state = SilentState("S", alphabet, False)
    hmm.add_state(start_state, LOG(1.0))

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state, LOG(0.0))

    M1 = NormalState("M1", {"A": LOG(0.8), "C": LOG(0.2), "G": LOG(0.0), "U": LOG(0.0)})
    hmm.add_state(M1, LOG(0.0))

    M2 = NormalState("M2", {"A": LOG(0.4), "C": LOG(0.6), "G": LOG(0.0), "U": LOG(0.6)})
    hmm.add_state(M2, LOG(0.0))

    hmm.set_trans("S", "M1", LOG(1.0))
    hmm.set_trans("M1", "M2", LOG(1.0))
    hmm.set_trans("M2", "E", LOG(1.0))
    hmm.normalize()

    random = RandomState(0)
    path = hmm.emit(random)
    assert [p[0] for p in path] == ["S", "M1", "M2", "E"]
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

    logp = hmm.likelihood("CG", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)], True)
    assert isinf(logp)

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

    with pytest.raises(ValueError):
        hmm.likelihood("UU", [("S", 0), ("M1", 1), ("M22", 1), ("E", 0)])


def test_hmm_lik_2():
    alphabet = "AC"

    hmm = HMM(alphabet)
    start_state = NormalState("S", {"A": LOG(0.8), "C": LOG(0.2)})
    hmm.add_state(start_state, LOG(1.0))

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state, LOG(0.0))

    hmm.set_trans("S", "E", LOG(1.0))
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

    logp = hmm.likelihood("", [], True)
    assert_allclose(logp, 0.0, atol=1e-7)

    p = hmm.likelihood("A", [])
    assert_allclose(p, 0.0)

    logp = hmm.likelihood("A", [], True)
    assert isinf(logp)


def test_hmm_lik_3():
    alphabet = "AC"

    hmm = HMM(alphabet)
    start_state = SilentState("S", alphabet, False)
    hmm.add_state(start_state, LOG(1.0))

    M1 = SilentState("M1", alphabet, False)
    hmm.add_state(M1, LOG(0.0))

    M2 = NormalState("M2", {"A": LOG(0.8), "C": LOG(0.2)})
    hmm.add_state(M2, LOG(0.0))

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state, LOG(0.0))

    hmm.set_trans("S", "M1", LOG(1.0))
    hmm.set_trans("M1", "M2", LOG(1.0))
    hmm.set_trans("M2", "E", LOG(1.0))
    hmm.normalize()

    p = hmm.likelihood("A", [("S", 1), ("E", 0)])
    assert_allclose(p, 0.0, atol=1e-7)

    p = hmm.likelihood("A", [("S", 0), ("M1", 0), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.8)

    p = hmm.likelihood("C", [("S", 0), ("M1", 0), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.2)

    p = hmm.likelihood("C", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.0)

    hmm.set_trans("M1", "E", LOG(1.0))
    hmm.normalize()

    p = hmm.likelihood("A", [("S", 0), ("M1", 0), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.4)

    p = hmm.likelihood("C", [("S", 0), ("M1", 0), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.1)

    states_path = [("S", 0), ("M1", 0), ("E", 0)]
    p = hmm.likelihood("", states_path)
    assert_allclose(p, 0.5)
    assert states_path == [("S", 0), ("M1", 0), ("E", 0)]


def test_hmm_viterbi_1():
    alphabet = "ACGU"

    hmm = HMM(alphabet)
    start_state = SilentState("S", alphabet, False)
    hmm.add_state(start_state, LOG(1.0))

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state, LOG(0.0))

    M1 = NormalState("M1", {"A": LOG(0.8), "C": LOG(0.2), "G": LOG(0.0), "U": LOG(0.0)})
    hmm.add_state(M1, LOG(0.0))

    M2 = NormalState("M2", {"A": LOG(0.4), "C": LOG(0.6), "G": LOG(0.0), "U": LOG(0.6)})
    hmm.add_state(M2, LOG(0.0))

    hmm.set_trans("S", "M1", LOG(1.0))
    hmm.set_trans("M1", "M2", LOG(1.0))
    hmm.set_trans("M2", "E", LOG(1.0))
    hmm.normalize()

    random = RandomState(0)
    path = hmm.emit(random)
    assert [p[0] for p in path] == ["S", "M1", "M2", "E"]
    assert "".join(p[1] for p in path) == "AC"

    lik, path = hmm.viterbi("AC", "E")
    assert_allclose(lik, 0.3)
    assert [p[0] for p in path] == ["S", "M1", "M2", "E"]
    assert path[0][1] == 0
    assert path[1][1] == 1
    assert path[2][1] == 1
    assert path[3][1] == 0

    lik = hmm.viterbi("AC", "E", True)[0]
    assert_allclose(lik, LOG(0.3))

    p = hmm.likelihood("AC", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.3)
    logp = hmm.likelihood("AC", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)], True)
    assert_allclose(logp, LOG(0.3))


def test_hmm_viterbi_2():
    alphabet = "AC"

    hmm = HMM(alphabet)
    start_state = SilentState("S", alphabet, False)
    hmm.add_state(start_state, LOG(1.0))

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state, LOG(0.0))

    M1 = NormalState("M1", {"A": LOG(0.8), "C": LOG(0.2)})
    hmm.add_state(M1, LOG(0.0))

    M2 = NormalState("M2", {"A": LOG(0.4), "C": LOG(0.6)})
    hmm.add_state(M2, LOG(0.0))

    hmm.set_trans("S", "M1", LOG(1.0))
    hmm.set_trans("M1", "M2", LOG(1.0))
    hmm.set_trans("M2", "E", LOG(1.0))
    hmm.normalize()

    lik, path = hmm.viterbi("AC", "E")
    assert_allclose(lik, 0.48)
    assert [p[0] for p in path] == ["S", "M1", "M2", "E"]
    assert list(p[1] for p in path) == [0, 1, 1, 0]
    p = hmm.likelihood("AC", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.48)

    lik, path = hmm.viterbi("AA", "E")
    assert_allclose(lik, 0.32)
    assert [p[0] for p in path] == ["S", "M1", "M2", "E"]
    assert list(p[1] for p in path) == [0, 1, 1, 0]
    p = hmm.likelihood("AA", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.32)

    lik, path = hmm.viterbi("CA", "E")
    assert_allclose(lik, 0.08)
    assert [p[0] for p in path] == ["S", "M1", "M2", "E"]
    assert list(p[1] for p in path) == [0, 1, 1, 0]
    p = hmm.likelihood("CA", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.08)

    lik, path = hmm.viterbi("CC", "E")
    assert_allclose(lik, 0.12)
    assert [p[0] for p in path] == ["S", "M1", "M2", "E"]
    assert list(p[1] for p in path) == [0, 1, 1, 0]
    p = hmm.likelihood("CC", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.12)

    hmm.set_trans("M1", "E", LOG(1.0))
    hmm.normalize()

    lik, path = hmm.viterbi("AC", "E")
    assert_allclose(lik, 0.48)
    assert [p[0] for p in path] == ["S", "M1", "M2", "E"]
    assert list(p[1] for p in path) == [0, 1, 1, 0]
    p = hmm.likelihood("AC", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.24)

    lik, path = hmm.viterbi("AA", "E")
    assert_allclose(lik, 0.32)
    assert [p[0] for p in path] == ["S", "M1", "M2", "E"]
    assert list(p[1] for p in path) == [0, 1, 1, 0]
    p = hmm.likelihood("AA", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.16)

    lik, path = hmm.viterbi("CA", "E")
    assert_allclose(lik, 0.08)
    assert [p[0] for p in path] == ["S", "M1", "M2", "E"]
    assert list(p[1] for p in path) == [0, 1, 1, 0]
    p = hmm.likelihood("CA", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.04)

    lik, path = hmm.viterbi("CC", "E")
    assert_allclose(lik, 0.12)
    assert [p[0] for p in path] == ["S", "M1", "M2", "E"]
    assert list(p[1] for p in path) == [0, 1, 1, 0]
    p = hmm.likelihood("CC", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.06)


def test_hmm_viterbi_3():
    alphabet = "AC"

    hmm = HMM(alphabet)
    start_state = SilentState("S", alphabet, False)
    hmm.add_state(start_state, LOG(1.0))

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state, LOG(0.0))

    M1 = NormalState("M1", {"A": LOG(0.8), "C": LOG(0.2)})
    hmm.add_state(M1, LOG(0.0))

    D1 = SilentState("D1", alphabet, False)
    hmm.add_state(D1, LOG(0.0))

    M2 = NormalState("M2", {"A": LOG(0.4), "C": LOG(0.6)})
    hmm.add_state(M2, LOG(0.0))

    D2 = SilentState("D2", alphabet, False)
    hmm.add_state(D2, LOG(0.0))

    hmm.set_trans("S", "M1", LOG(0.8))
    hmm.set_trans("S", "D1", LOG(0.2))

    hmm.set_trans("M1", "M2", LOG(0.8))
    hmm.set_trans("M1", "D2", LOG(0.2))

    hmm.set_trans("D1", "D2", LOG(0.2))
    hmm.set_trans("D1", "M2", LOG(0.8))

    hmm.set_trans("D2", "E", LOG(1.0))
    hmm.set_trans("M2", "E", LOG(1.0))
    hmm.normalize()

    lik, path = hmm.viterbi("AC", "E")
    assert_allclose(lik, 0.3072)
    assert [p[0] for p in path] == ["S", "M1", "M2", "E"]
    assert list(p[1] for p in path) == [0, 1, 1, 0]
    p = hmm.likelihood("AC", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.3072)

    lik, path = hmm.viterbi("AA", "E")
    assert_allclose(lik, 0.2048)
    assert [p[0] for p in path] == ["S", "M1", "M2", "E"]
    assert list(p[1] for p in path) == [0, 1, 1, 0]
    p = hmm.likelihood("AA", [("S", 0), ("M1", 1), ("M2", 1), ("E", 0)])
    assert_allclose(p, 0.2048)

    lik, path = hmm.viterbi("A", "E")
    assert_allclose(lik, 0.128)
    assert [p[0] for p in path] == ["S", "M1", "D2", "E"]
    assert list(p[1] for p in path) == [0, 1, 0, 0]
    p = hmm.likelihood("A", [("S", 0), ("M1", 1), ("D2", 0), ("E", 0)])
    assert_allclose(p, 0.128)


def test_hmm_draw(tmp_path):
    hmm = _create_hmm()

    base_emission = {"A": LOG(0.5), "C": LOG(0.5)}
    codon_emission = {"ACC": LOG(0.8), "AAA": LOG(0.2)}
    epsilon = 0.1
    M0 = FrameState("M0", base_emission, codon_emission, epsilon)
    hmm.add_state(M0, LOG(1.0))
    hmm.set_trans("M0", "E", LOG(1.0))
    hmm.normalize()

    hmm.draw(tmp_path / "test.pdf", emissions=5, init_prob=True)
    hmm.draw(tmp_path / "test.pdf", emissions=3, init_prob=False)
    hmm.draw(tmp_path / "test.pdf", emissions=0, init_prob=True)
    hmm.draw(tmp_path / "test.pdf", emissions=0, init_prob=False)
    hmm.draw(tmp_path / "test.pdf", emissions=50, init_prob=True)


def test_hmm_rename_state():
    hmm = _create_hmm()

    with pytest.raises(ValueError):
        hmm.rename_state("DD", "D1")

    with pytest.raises(ValueError):
        hmm.rename_state("E", "M1")

    p = hmm.trans("D1", "D2")
    hmm.rename_state("S", "B")
    hmm.rename_state("D2", "DD")
    assert "B" in hmm.states
    assert "S" not in hmm.states
    assert "DD" in hmm.states
    assert "D2" not in hmm.states
    assert hmm.trans("D1", "DD") == p


def test_hmm_delete_state():
    hmm = _create_hmm()

    with pytest.raises(ValueError):
        hmm.delete_state("D22")

    hmm.delete_state("D2")
    assert "D2" not in hmm.states


def test_hmm_single_state():
    alphabet = "ACGU"
    hmm = HMM(alphabet)
    state = NormalState(
        "I", {"A": LOG(0.8), "C": LOG(0.2), "G": LOG(0.0), "U": LOG(0.0)}
    )
    hmm.add_state(state)

    hmm.normalize()
    lik, path = hmm.viterbi("ACC", "I")
    assert abs(lik - 0.032) < 1e-7
    assert path == [("I", 1), ("I", 1), ("I", 1)]


def _create_hmm():
    alphabet = "AC"

    hmm = HMM(alphabet)
    start_state = SilentState("S", alphabet, False)
    hmm.add_state(start_state, LOG(1.0))

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state, LOG(0.0))

    M1 = NormalState("M1", {"A": LOG(0.8), "C": LOG(0.2)})
    hmm.add_state(M1, LOG(0.0))

    D1 = SilentState("D1", alphabet, False)
    hmm.add_state(D1, LOG(0.0))

    M2 = NormalState("M2", {"A": LOG(0.4), "C": LOG(0.6)})
    hmm.add_state(M2, LOG(0.0))

    D2 = SilentState("D2", alphabet, False)
    hmm.add_state(D2, LOG(0.0))

    hmm.set_trans("S", "M1", LOG(0.8))
    hmm.set_trans("S", "D1", LOG(0.2))

    hmm.set_trans("M1", "M2", LOG(0.8))
    hmm.set_trans("M1", "D2", LOG(0.2))

    hmm.set_trans("D1", "D2", LOG(0.2))
    hmm.set_trans("D1", "M2", LOG(0.8182787382))

    hmm.set_trans("D2", "E", LOG(1.0))
    hmm.set_trans("M2", "E", LOG(1.0))

    return hmm

