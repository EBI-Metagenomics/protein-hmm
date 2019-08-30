from math import log
from numpy.testing import assert_allclose
from numpy.random import RandomState
from hseq import SilentState, NormalState, TripletState, FrameState


def test_states():
    start_state = SilentState("S", False)
    assert start_state.name == "S"
    assert start_state.end_state is False
    assert start_state.emit(RandomState(0)) == ""
    assert start_state.alphabet == ""
    assert_allclose(start_state.prob(""), 1.0)
    assert_allclose(start_state.prob("A"), 0.0, atol=1e-7)

    end_state = SilentState("E", True)
    assert end_state.name == "E"
    assert end_state.end_state is True
    assert end_state.emit(RandomState(0)) == ""
    assert start_state.alphabet == ""

    normal_state = NormalState("M1", {"A": -log(0.99), "B": -log(0.01)})
    assert normal_state.name == "M1"
    assert normal_state.end_state is False
    assert normal_state.emit(RandomState(0)) == "A"
    assert normal_state.emit(RandomState(1)) == "A"
    assert normal_state.emit(RandomState(2)) == "A"
    assert normal_state.emit(RandomState(3)) == "A"
    assert set(normal_state.alphabet) == set("AB")
    assert_allclose(normal_state.prob("A"), 0.99)
    assert_allclose(normal_state.prob("B"), 0.01)

    alphabet = "ACGU"
    triplet_state = TripletState("M2", alphabet, {"AUG": -log(0.8), "AUU": -log(0.8)})
    assert triplet_state.name == "M2"
    assert triplet_state.end_state is False
    assert triplet_state.emit(RandomState(0)) == "AUU"
    assert triplet_state.emit(RandomState(1)) == "AUG"
    assert set(triplet_state.alphabet) == set("ACGU")
    assert_allclose(triplet_state.prob("AUG"), 0.5)
    assert_allclose(triplet_state.prob("AUU"), 0.5)
    assert_allclose(triplet_state.prob("AGU"), 0.0, atol=1e-7)

    codon_emission = {"AUG": -log(0.8), "AUU": -log(0.1)}
    frame_state = FrameState("M3", "ACGU", codon_emission, epsilon=0.1)
    assert_allclose(frame_state._codon_prob("A", "U", "G"), 0.8888888888888888)
    assert_allclose(frame_state._codon_prob("A", "U", "U"), 0.11111111111111115)
    assert_allclose(frame_state._codon_prob("A", "U", None), 1.0)
    assert_allclose(frame_state._codon_prob("A", None, "U"), 0.11111111111111115)
    assert_allclose(frame_state._codon_prob(None, "G", "U"), 0.0)
    assert_allclose(frame_state._codon_prob(None, "U", "U"), 0.11111111111111115)
    assert_allclose(frame_state._codon_prob(None, None, "U"), 0.11111111111111115)
    assert_allclose(frame_state._codon_prob(None, None, None), 1.0)


