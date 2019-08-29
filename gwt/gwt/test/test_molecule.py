from gwt import RNA, DNA


def test_molecule():
    rna = RNA()
    assert set(rna.bases) == set("ACGU")

    dna = DNA()
    assert set(dna.bases) == set("ACGT")
