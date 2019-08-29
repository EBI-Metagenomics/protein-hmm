from math import log
from gwt import AA2Codon
from gwt import DNA


def test_aa2codon():
    aa_emission = {"A": -log(0.2)}
    aa2codon = AA2Codon(aa_emission, DNA())
    assert set(aa2codon.bases) == set(DNA().bases)
    assert aa2codon.amino_acids == "A"
    assert aa2codon.codon_emission() == {
        "GCT": 0.25,
        "GCC": 0.25,
        "GCA": 0.25,
        "GCG": 0.25,
    }
    assert aa2codon.aa_emission() == {"A": 1.0}
