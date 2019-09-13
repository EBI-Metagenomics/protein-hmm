from math import log

from numpy.testing import assert_allclose

from gwt import DNA, RNA, FrameEmission, AA2Codon


def test_frame():

    codon_emission = {"UCU": log(0.9), "GUA": log(0.1)}
    fe = FrameEmission(codon_emission, RNA(), 0.0)

    assert fe.len_prob(1) == 0.0
    assert fe.indel_prob(1) == 0.0

    assert_allclose(fe.prob(list("UCU")), 0.9)
    assert_allclose(fe.prob(list("GUA")), 0.1)
    assert_allclose(fe.prob(list("GGA")), 0.0)

    assert_allclose(fe.prob("UCU"), 0.9)
    assert_allclose(fe.prob("GUA"), 0.1)
    assert_allclose(fe.prob("GGA"), 0.0)

    fe = FrameEmission(codon_emission, RNA(), 1e-2)

    assert abs(fe.len_prob(1) - 9.801e-05) < 1e-4
    assert abs(fe.indel_prob(1) - 0.03881196) < 1e-5

    assert abs(fe.prob("UCU") - 0.864565812326389) < 1e-5
    assert abs(fe.prob("GUA") - 0.09606286814583331) < 1e-5
    assert abs(fe.prob("GGA") - 2.178020833333333e-06) < 1e-5

    fe = FrameEmission(codon_emission, RNA(), 0.5)

    assert abs(fe.len_prob(1) - 0.0625) < 1e-4
    assert abs(fe.indel_prob(1) - 0.25) < 1e-5

    assert abs(fe.prob("UCU") - 0.07703993055555558) < 1e-5
    assert abs(fe.prob("GUA") - 0.009244791666666667) < 1e-5
    assert abs(fe.prob("GGA") - 0.0015190972222222225) < 1e-5

    assert set(fe.bases) == set(RNA().bases)

    codon_emission = {"TCT": log(0.9), "GTA": log(0.1)}
    fe = FrameEmission(codon_emission, DNA(), 0.5)

    assert abs(fe.len_prob(1) - 0.0625) < 1e-4
    assert abs(fe.indel_prob(1) - 0.25) < 1e-5

    assert abs(fe.prob("TCT") - 0.07703993055555558) < 1e-5
    assert abs(fe.prob("GTA") - 0.009244791666666667) < 1e-5
    assert abs(fe.prob("GGA") - 0.0015190972222222225) < 1e-5

    assert set(fe.bases) == set(DNA().bases)

    assert fe.len_prob(0) == 0.0
    assert fe.len_prob(6) == 0.0

    assert fe.indel_prob(5) == 0.0

    fe = FrameEmission(codon_emission, DNA(), 0.1)
    output = str(fe)
    assert "Epsilon = 0.1" in output
    assert "p(X=TCT) = 0.9000" in output
    assert "p(Z=C | F=1) = 0.3000" in output
    assert "p(Z=TCT  , F=3) = 0.5929" in output

    compo = {"A": log(0.7), "C": log(0.1), "G": log(0.1), "T": log(0.1)}
    fe = FrameEmission(codon_emission, DNA(), 0.5, compo)

    assert abs(fe.len_prob(1) - 0.0625) < 1e-4
    assert abs(fe.indel_prob(1) - 0.25) < 1e-5

    assert abs(fe.prob("TCT") - 0.06407638888888889) < 1e-5
    assert abs(fe.prob("GTA") - 0.009729166666666667) < 1e-5
    assert abs(fe.prob("GGA") - 0.0006597222222222225) < 1e-5


def test_frame_emission():
    aa_emission = {
        "K": -1.918460000000000054,
        "R": -2.154889999999999972,
        "E": -2.307549999999999990,
        "S": -2.578129999999999811,
        "Q": -2.579489999999999839,
        "T": -2.649999999999999911,
        "A": -2.759770000000000056,
        "D": -2.822480000000000100,
        "N": -2.840310000000000112,
        "P": -2.971490000000000187,
        "H": -3.169970000000000176,
        "G": -3.543489999999999807,
        "L": -3.608680000000000110,
        "V": -3.694659999999999833,
        "I": -4.133429999999999715,
        "Y": -4.327270000000000394,
        "M": -4.334719999999999906,
        "F": -4.648749999999999716,
        "C": -5.304280000000000328,
        "W": -5.736670000000000158,
    }
    molecule = DNA()
    aa2codon = AA2Codon(aa_emission, molecule)
    epsilon = 0.1
    frame = FrameEmission(aa2codon.codon_emission(True), molecule, epsilon)
    emission = frame.emission()
    assert emission[0][0] == "AAA"
    assert_allclose(emission[0][1], 0.04947252553677521)
    assert emission[30][0] == "GCT"
    assert_allclose(emission[30][1], 0.010766873439039608)
