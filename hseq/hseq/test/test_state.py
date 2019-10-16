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
    assert frame_state.min_len == 1
    assert frame_state.max_len == 5

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


def test_state_frame_base_emiss1():
    base_emission = {"A": log(0.1), "C": log(0.2), "G": log(0.3), "U": log(0.4)}
    codon_emission = {"AUG": log(0.8), "AUU": log(0.1)}
    epsilon = 0.1
    frame_state = FrameState("M5", base_emission, codon_emission, epsilon)
    assert_allclose(frame_state.prob("A", True), -5.914503505971854)
    assert_allclose(frame_state.prob("C"), 0.0)
    assert_allclose(frame_state.prob("G", True), -6.032286541628237)
    assert_allclose(frame_state.prob("U", True), -5.809142990314027)
    assert_allclose(frame_state.prob("AU", True), -2.9159357500274385)
    assert_allclose(frame_state.prob("AUA", True), -7.821518343902165)
    assert_allclose(frame_state.prob("AUG", True), -0.5344319079005616)
    assert_allclose(frame_state.prob("AUU", True), -2.57514520832882)
    assert_allclose(frame_state.prob("AUC", True), -7.129480084106424)
    assert_allclose(frame_state.prob("AUUA", True), -7.789644584138959)
    assert_allclose(frame_state.prob("ACUG", True), -5.036637096635257)
    assert_allclose(frame_state.prob("AUUAA", True), -13.920871073622099)
    assert_allclose(frame_state.prob("AUUAAA"), 0.0)


def test_state_frame_base_emiss2():
    base_emission = {"A": log(0.1), "C": log(0.2), "G": log(0.3), "T": log(0.4)}
    codon_emission = {"ATG": log(0.8), "ATT": log(0.1), "GTC": log(0.4)}
    epsilon = 0.1
    frame_state = FrameState("M5", base_emission, codon_emission, epsilon)
    assert_allclose(frame_state.prob("A", True), -6.282228286097171)
    assert_allclose(frame_state.prob("C", True), -7.0931585023135)
    assert_allclose(frame_state.prob("G", True), -5.99454621364539)
    assert_allclose(frame_state.prob("T", True), -5.840395533818132)
    assert_allclose(frame_state.prob("AT", True), -3.283414346005771)
    assert_allclose(frame_state.prob("CG", True), -9.395743595307545)
    assert_allclose(frame_state.prob("ATA", True), -8.18911998648269)
    assert_allclose(frame_state.prob("ATG", True), -0.9021560981322401)
    assert_allclose(frame_state.prob("ATT", True), -2.9428648000333952)
    assert_allclose(frame_state.prob("ATC", True), -7.314811395663229)
    assert_allclose(frame_state.prob("GTC", True), -1.5951613351178675)
    assert_allclose(frame_state.prob("ATTA", True), -8.157369364264277)
    assert_allclose(frame_state.prob("GTTC", True), -4.711642430498609)
    assert_allclose(frame_state.prob("ACTG", True), -5.404361876760574)
    assert_allclose(frame_state.prob("ATTAA", True), -14.288595853747417)
    assert_allclose(frame_state.prob("GTCAA", True), -12.902301492627526)
    assert_allclose(frame_state.prob("ATTAAA"), 0.0)
