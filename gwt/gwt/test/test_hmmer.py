import importlib_resources as pkg_resources
from numpy.random import RandomState

import gwt


def test_read_hmmer(tmp_path):
    text = pkg_resources.read_text(gwt.test, "PF02545.hmm")

    with open(tmp_path / "PF02545.hmm", "w") as f:
        f.write(text)

    hmm = gwt.read_hmmer(tmp_path / "PF02545.hmm")
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
