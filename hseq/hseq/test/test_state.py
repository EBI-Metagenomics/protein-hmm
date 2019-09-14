from itertools import product
from math import log, exp

from numpy.random import RandomState
from numpy.testing import assert_allclose

from hseq import FrameState, NormalState, SilentState, TripletState, LOG


def test_states():
    start_state = SilentState("S", "ACGU", False)
    assert start_state.name == "S"
    assert start_state.end_state is False
    assert start_state.emit(RandomState(0)) == ""
    assert start_state.alphabet == "ACGU"
    assert_allclose(start_state.prob(""), 1.0)
    assert_allclose(start_state.prob("A"), 0.0, atol=1e-7)
    assert str(start_state) == "<S>"
    assert repr(start_state) == "<SilentState:S>"

    table = start_state.emission()
    assert len(table) == 1
    assert table[0][0] == ""
    assert table[0][1] == 1.0

    table = start_state.emission(log_space=True)
    assert len(table) == 1
    assert table[0][0] == ""
    assert table[0][1] == LOG(1.0)

    end_state = SilentState("E", "ACGU", True)
    assert end_state.name == "E"
    assert end_state.end_state is True
    assert end_state.emit(RandomState(0)) == ""
    assert start_state.alphabet == "ACGU"

    normal_state = NormalState("M1", {"A": log(0.99), "B": log(0.01)})
    assert normal_state.name == "M1"
    assert normal_state.end_state is False
    assert normal_state.emit(RandomState(0)) == "A"
    assert normal_state.emit(RandomState(1)) == "A"
    assert normal_state.emit(RandomState(2)) == "A"
    assert normal_state.emit(RandomState(3)) == "A"
    assert set(normal_state.alphabet) == set("AB")
    assert_allclose(normal_state.prob("A"), 0.99)
    assert_allclose(normal_state.prob("B"), 0.01)
    assert str(normal_state) == "<M1>"
    assert repr(normal_state) == "<NormalState:M1>"

    table = normal_state.emission(log_space=False)
    assert table[0][0] == "A"
    assert_allclose(table[0][1], 0.99)
    assert table[1][0] == "B"
    assert_allclose(table[1][1], 0.01)

    alphabet = "ACGU"
    triplet_state = TripletState("M2", alphabet, {"AUG": log(0.8), "AUU": log(0.8)})
    assert triplet_state.name == "M2"
    assert triplet_state.end_state is False
    assert triplet_state.emit(RandomState(0)) == "AUU"
    assert triplet_state.emit(RandomState(1)) == "AUG"
    assert set(triplet_state.alphabet) == set("ACGU")
    assert_allclose(triplet_state.prob("AUG"), 0.5)
    assert_allclose(triplet_state.prob("AUU"), 0.5)
    assert_allclose(triplet_state.prob("AGU"), 0.0, atol=1e-7)
    assert str(triplet_state) == "<M2>"
    assert repr(triplet_state) == "<TripletState:M2>"

    table = triplet_state.emission(log_space=False)
    assert table[0][0] == "AUG"
    assert_allclose(table[0][1], 0.5)
    assert table[1][0] == "AUU"
    assert_allclose(table[1][1], 0.5)

    assert triplet_state.min_len == 3
    assert triplet_state.max_len == 3

    base_emission = {"A": log(0.25), "C": log(0.25), "G": log(0.25), "U": log(0.25)}
    codon_emission = {"AUG": log(0.8), "AUU": log(0.1)}
    epsilon = 0.1
    frame_state = FrameState("M3", base_emission, codon_emission, epsilon)
    assert_allclose(frame_state._codon_prob("A", "U", "G"), LOG(0.8888888888888888))
    assert_allclose(frame_state._codon_prob("A", "U", "U"), LOG(0.11111111111111115))
    assert_allclose(frame_state._codon_prob("A", "U", None), LOG(1.0))
    assert_allclose(frame_state._codon_prob("A", None, "U"), LOG(0.11111111111111115))
    assert_allclose(frame_state._codon_prob(None, "G", "U"), LOG(0.0))
    assert_allclose(frame_state._codon_prob(None, "U", "U"), LOG(0.11111111111111115))
    assert_allclose(frame_state._codon_prob(None, None, "U"), LOG(0.11111111111111115))
    assert_allclose(frame_state._codon_prob(None, None, None), LOG(1.0))

    table = frame_state.emission(log_space=True)
    assert table[0][0] == "AUG"
    assert_allclose(table[0][1], -0.5347732882047063)
    assert table[1][0] == "AUU"
    assert_allclose(table[1][1], -2.590237330499946)
    assert table[2][0] == "AU"
    assert_allclose(table[2][1], -2.915843423869834)

    epsilon = 0.0
    frame_state = FrameState("M4", base_emission, codon_emission, epsilon)
    assert_allclose(frame_state.prob("AUA"), 0.0)
    assert_allclose(frame_state.prob("AUG"), 0.8888888888888888)
    assert_allclose(frame_state.prob("AUU"), 0.11111111111111115)
    assert_allclose(frame_state.prob("AU"), 0.0)
    assert_allclose(frame_state.prob("A"), 0.0)
    assert_allclose(frame_state.prob("AUUA"), 0.0)
    assert_allclose(frame_state.prob("AUUAA"), 0.0)

    epsilon = 0.1
    frame_state = FrameState("M5", base_emission, codon_emission, epsilon)
    assert_allclose(frame_state.prob("AUA"), 0.0010021604938271608)
    assert_allclose(frame_state.prob("AUG"), 0.5858020833333333)
    assert_allclose(frame_state.prob("AUU"), 0.07500223765432103)
    assert_allclose(frame_state.prob("AU"), 0.054158333333333336)
    assert_allclose(frame_state.prob("A"), 0.0027000000000000006)
    assert_allclose(frame_state.prob("AUUA"), 0.0010270833333333336)
    assert_allclose(frame_state.prob("AUUAA"), 5.625000000000003e-06, atol=1e-5)
    assert_allclose(frame_state.prob("AUUAAA"), 0.0, atol=1e-7)
    assert str(frame_state) == "<M5>"
    assert repr(frame_state) == "<FrameState:M5>"

    p = frame_state._prob_z_given_f
    abc = "ACGU"
    for f in range(1, 6):
        assert_allclose(sum(exp(p(x)) for x in product(*[abc] * f)), 1.0)

    random = RandomState(0)
    assert frame_state.emit(random) == "AUG"
    assert frame_state.emit(random) == "AUG"
    assert frame_state.emit(random) == "AUG"
    assert frame_state.emit(random) == "AUU"
    assert frame_state.emit(random) == "AUGA"
    assert frame_state.emit(random) == "AUG"
    assert frame_state.emit(random) == "AUU"
    assert frame_state.emit(random) == "AG"
    assert frame_state.emit(random) == "UG"
