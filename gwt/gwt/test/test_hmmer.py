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

    assert hmm.init_prob("S") == 1.0
    assert hmm.init_prob("E") == 0.0
    assert "I166" in hmm.states
    assert "ACDEFGHIKLMNPQRSTVWY" == hmm.alphabet
    assert abs(hmm.trans("M2", "D3") - 0.0036089723618955506) < 1e-7
    assert hmm.trans("M2", "D2") < 1e-7

    random = RandomState(0)
    path = hmm.emit(random)
    seq = "".join(i[1] for i in path)
    states = "".join(str(i[0]) for i in path)
    assert (
        seq
        == "NLVLASASSSRQTLLNQMKARDKIDLLEPESVYWRIAHAKIMTREAAGVNNVSGKNQLPPFILIGMDNVVVYTLRKAKTSEDAAEVCQEMQGEVIELTGALVFGVKSTSVFRFAKLNDDKELVRLVFAQGAWLGVQFMKVKFSKAYVELDQRCNKSGAIPIEASGGEAFEVAKGDYTNTLGLPGVNLNTELKSW"
    )
    assert len(states) == 1068


def test_read_hmmer_2(tmp_path):
    text = pkg_resources.read_text(gwt.test, "PF03373.hmm")

    with open(tmp_path / "PF03373.hmm", "w") as f:
        f.write(text)

    hmmfile = gwt.read_hmmer_file(tmp_path / "PF03373.hmm")
    hmm = gwt.create_hmmer_profile(hmmfile)

    random = RandomState(0)
    path = hmm.emit(random)
    seq = "".join(i[1] for i in path)
    states = "".join(str(i[0]) for i in path)
    assert abs(hmm.trans("M0", "M1") - 0.9892313034087644) < 1e-7
    assert seq == "PKREDRKM"
    assert len(states) == 42


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
    states = "".join(str(i[0]) for i in path)
    assert abs(hmm.trans("M0", "M1") - 0.9892313034087644) < 1e-7
    assert seq == "CCCGGUGAGGAGAAUGGGAAUGAA"
    assert len(states) == 42

    hmm = gwt.create_frame_hmm(hmmfile, phmm, 0.5)

    random = RandomState(0)
    path = hmm.emit(random)
    seq = "".join(i[1] for i in path)
    states = "".join(str(i[0]) for i in path)
    assert abs(hmm.trans("M0", "M1") - 0.9892313034087644) < 1e-7
    assert seq == "CCCGGUGAGGGUAAGGGGAAUUGA"
    assert len(states) == 42


def test_create_frame_hmm_exceptions(tmp_path):
    with pytest.raises(ValueError):
        hmmfile = gwt.read_hmmer_file(tmp_path)

    with pytest.raises(ValueError):
        hmmfile = gwt.read_hmmer_file(tmp_path / "this_doesnt_exist.file")
