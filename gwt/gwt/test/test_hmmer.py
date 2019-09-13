import pytest
import importlib_resources as pkg_resources
from numpy.random import RandomState

import gwt


def test_read_hmmer_1(tmp_path):
    text = pkg_resources.read_text(gwt.test, "PF02545.hmm")

    with open(tmp_path / "PF02545.hmm", "w") as f:
        f.write(text)

    hmmfile = gwt.read_hmmer_file(tmp_path / "PF02545.hmm")
    hmm = gwt.create_hmmer_profile(hmmfile)

    assert hmm.init_prob("B") == 1.0
    assert hmm.init_prob("E") == 0.0
    assert "I166" in hmm.states
    assert "ACDEFGHIKLMNPQRSTVWY" == hmm.alphabet
    assert abs(hmm.trans("M2", "D3") - 0.0036089723618955506) < 1e-7
    assert hmm.trans("M2", "D2") < 1e-7

    random = RandomState(0)
    path = hmm.emit(random)
    seq = "".join(i[1] for i in path)
    states = [i[0] for i in path]
    iseq = "PLKVHSAARYRDDLLKAMVIPQIIPYDQGEPESVYWRIAHAKIMTREAAGVNNVSGKNQLP"
    iseq += "PFILIGMDNVVVYTLRKAKTSEDAAEVCQEMQGEVIELTGALVFGVKSTSVFRFAKLNDD"
    iseq += "KELVRLVFAQGAWLGVQFMKVKFSKAYVELDQRCNKSGAIPIEASGGEAFEVAKGDYTNT"
    iseq += "LGLPGVNLNTELKSW"
    assert seq == iseq
    assert len(states) == 201


def test_read_hmmer_2(tmp_path):
    text = pkg_resources.read_text(gwt.test, "PF03373.hmm")
    with open(tmp_path / "PF03373.hmm", "w") as f:
        f.write(text)

    hmmfile = gwt.read_hmmer_file(tmp_path / "PF03373.hmm")
    hmm = gwt.create_hmmer_profile(hmmfile)

    random = RandomState(0)
    path = hmm.emit(random)
    seq = "".join(i[1] for i in path)
    states = [i[0] for i in path]
    assert abs(hmm.trans("B", "M1") - 0.9892313034087644) < 1e-7
    assert seq == "PGKRDNGA"
    assert len(states) == 10


def test_create_frame_hmm(tmp_path):
    text = pkg_resources.read_text(gwt.test, "PF03373.hmm")
    with open(tmp_path / "PF03373.hmm", "w") as f:
        f.write(text)

    hmmfile = gwt.read_hmmer_file(tmp_path / "PF03373.hmm")
    phmm = gwt.create_hmmer_profile(hmmfile)
    hmm = gwt.create_frame_hmm(hmmfile, phmm, 1e-3)

    random = RandomState(0)
    path = hmm.emit(random)
    seq = "".join(i[1] for i in path)
    states = [i[0] for i in path]
    assert abs(hmm.trans("B", "M1") - 0.9892313034087644) < 1e-7
    assert seq == "CCGGAAAAGGAGGACGGCAAUAAA"
    assert len(states) == 10

    hmm = gwt.create_frame_hmm(hmmfile, phmm, 0.5)

    random = RandomState(0)
    path = hmm.emit(random)
    seq = "".join(i[1] for i in path)
    states = [i[0] for i in path]
    assert abs(hmm.trans("B", "M1") - 0.9892313034087644) < 1e-7
    assert seq == "CCGCCUAGAGAGAAACGGAAGAAACAAAG"
    assert len(states) == 10


def test_create_frame_hmm_exceptions(tmp_path):
    with pytest.raises(ValueError):
        gwt.read_hmmer_file(tmp_path)

    with pytest.raises(ValueError):
        gwt.read_hmmer_file(tmp_path / "this_doesnt_exist.file")


def test_create_frame_hmm_likelihood(tmp_path):
    text = pkg_resources.read_text(gwt.test, "PF03373.hmm")
    with open(tmp_path / "PF03373.hmm", "w") as f:
        f.write(text)

    hmmfile = gwt.read_hmmer_file(str(tmp_path / "PF03373.hmm"))
    phmm = gwt.create_hmmer_profile(hmmfile)
    hmm = gwt.create_frame_hmm(hmmfile, phmm, 0.0)

    states_path = [("B", 0)] + [(f"M{i}", 1) for i in range(1, 9)] + [("E", 0)]

    most_likely_seq = "PGKEDNNK"
    lik = phmm.likelihood(most_likely_seq, states_path)
    assert abs(lik - 0.02003508133944584) < 1e-7
    (lik, path) = phmm.viterbi(most_likely_seq)
    assert abs(lik - 0.02003508133944584) < 1e-7
    assert len(path) == 10

    states_path = [
        ("B", 0),
        ("M1", 1),
        ("M2", 1),
        ("M3", 1),
        ("M4", 1),
        ("M5", 1),
        ("M6", 1),
        ("M7", 1),
        ("I7", 1),
        ("M8", 1),
        ("E", 0),
    ]
    seq = "PGKEDNNSQ"
    assert abs(phmm.likelihood(seq, states_path) - 4.004833899558836e-07) < 1e-6
