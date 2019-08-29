from gwt import FrameEmission, RNA, DNA


def test_frame():

    codon_emission = {"UCU": 0.9, "GUA": 0.1}
    fe = FrameEmission(codon_emission, RNA(), 0.0)

    assert fe.len_prob(1) == 0.0
    assert fe.indel_prob(1) == 0.0

    assert fe.prob(list("UCU")) == 0.9
    assert fe.prob(list("GUA")) == 0.1
    assert fe.prob(list("GGA")) == 0.0

    assert fe.prob("UCU") == 0.9
    assert fe.prob("GUA") == 0.1
    assert fe.prob("GGA") == 0.0

    codon_emission = {"UCU": 0.9, "GUA": 0.1}
    fe = FrameEmission(codon_emission, RNA(), 1e-2)

    assert abs(fe.len_prob(1) - 9.801e-05) < 1e-4
    assert abs(fe.indel_prob(1) - 0.03881196) < 1e-5

    assert abs(fe.prob("UCU") - 0.864565812326389) < 1e-5
    assert abs(fe.prob("GUA") - 0.09606286814583331) < 1e-5
    assert abs(fe.prob("GGA") - 2.178020833333333e-06) < 1e-5

    codon_emission = {"UCU": 0.9, "GUA": 0.1}
    fe = FrameEmission(codon_emission, RNA(), 0.5)

    assert abs(fe.len_prob(1) - 0.0625) < 1e-4
    assert abs(fe.indel_prob(1) - 0.25) < 1e-5

    assert abs(fe.prob("UCU") - 0.07703993055555558) < 1e-5
    assert abs(fe.prob("GUA") - 0.009244791666666667) < 1e-5
    assert abs(fe.prob("GGA") - 0.0015190972222222225) < 1e-5

    assert set(fe.bases) == set(RNA().bases)

    codon_emission = {"TCT": 0.9, "GTA": 0.1}
    fe = FrameEmission(codon_emission, DNA(), 0.5)

    assert abs(fe.len_prob(1) - 0.0625) < 1e-4
    assert abs(fe.indel_prob(1) - 0.25) < 1e-5

    assert abs(fe.prob("TCT") - 0.07703993055555558) < 1e-5
    assert abs(fe.prob("GTA") - 0.009244791666666667) < 1e-5
    assert abs(fe.prob("GGA") - 0.0015190972222222225) < 1e-5

    assert set(fe.bases) == set(DNA().bases)
