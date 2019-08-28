from math import exp, log, inf
from itertools import product
from ._gencode import GENCODE


class AA2Codon:
    def __init__(self, aa_emission, gencode="standard", molecule="DNA"):
        self._gencode = GENCODE[gencode]
        convert_gencode_alphabet(self._gencode, molecule)

        if molecule == "DNA":
            self._bases = "ACGT"
        elif molecule == "RNA":
            self._bases = "ACGU"
        else:
            raise ValueError()

        normalize_emission(aa_emission)
        self._aa_emission = aa_emission

        self._codon_emission = {}
        self._generate_codon_emission()
        normalize_emission(self._codon_emission)

    @property
    def amino_acids(self):
        return "".join(sorted(self._aa_emission.keys()))

    @property
    def bases(self):
        return self._bases

    def aa_emission(self, prob_space=True):
        if prob_space:
            f = lambda x: exp(-x)
        else:
            f = lambda x: x
        return {k: f(v) for k, v in self._aa_emission.items()}

    def codon_emission(self, prob_space=True):
        if prob_space:
            f = lambda x: exp(-x)
        else:
            f = lambda x: x
        return {k: f(v) for k, v in self._codon_emission.items()}

    def _generate_codon_emission(self):
        for aa, nlogp in self._aa_emission.items():
            codons = self._gencode.get(aa, [])
            norm = log(len(codons))
            for codon in codons:
                self._codon_emission.update({codon: nlogp + norm})

        for a, b, c in product(*[self._bases] * 3):
            codon = a + b + c
            if codon not in self._codon_emission:
                self._codon_emission[codon] = inf


def infer_bases(amino_acids):
    bases = []
    for aa in amino_acids:
        bases += list(aa)
    return "".join(sorted(list(set(bases))))


def normalize_emission(emission):
    keys = list(emission.keys())
    nlogp = [emission[a] for a in keys]
    nlogp = normalize_nlogspace(nlogp)
    emission.update({a: logp for a, logp in zip(keys, nlogp)})


def normalize_nlogspace(values):
    from numpy import asarray
    from scipy.special import logsumexp

    values = asarray(values, float)
    norm = logsumexp(-values)
    return [norm + v for v in values]


def convert_gencode_alphabet(gencode, molecule):
    if molecule == "DNA":

        def replace(codon):
            return codon.replace("U", "T")

    elif molecule == "RNA":

        def replace(codon):
            return codon.replace("T", "U")

    for aa in gencode.keys():
        gencode[aa] = [replace(codon) for codon in gencode[aa]]

    return gencode
