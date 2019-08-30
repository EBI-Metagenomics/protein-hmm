from math import log
from numpy.random import RandomState
from hseq import SilentState, NormalState, TripletState


def test_states():
    start_state = SilentState("S", False)
    assert start_state.name == "S"
    assert start_state.end_state is False
    assert start_state.emit(RandomState(0)) == ""
    assert start_state.alphabet == ""

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

    alphabet = "ACGU"
    triplet_state = TripletState("M2", alphabet, {"AUG": -log(0.8), "AUU": -log(0.8)})
    assert triplet_state.name == "M2"
    assert triplet_state.end_state is False
    assert triplet_state.emit(RandomState(0)) == "AUU"
    assert triplet_state.emit(RandomState(1)) == "AUG"
    assert set(triplet_state.alphabet) == set("ACGU")
