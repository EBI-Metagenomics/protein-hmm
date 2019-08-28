from math import log
from gwt import AA2Codon


def test_aa2codon():
    aa_emission = {"A": -log(0.2)}
    aa2codon = AA2Codon(aa_emission)
    print(aa2codon.bases)
    print(aa2codon.amino_acids)
    print(aa2codon.aa_emission())
    print(aa2codon.codon_emission())
