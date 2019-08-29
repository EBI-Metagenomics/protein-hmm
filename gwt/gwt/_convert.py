from math import exp, log, inf
from ._gencode import GENCODE
from ._molecule import Molecule
from ._molecule import convert_to


class AA2Codon:
    def __init__(self, aa_emission, molecule: Molecule, gencode="standard"):
        self._gencode = GENCODE[gencode]
        self._molecule = molecule
        self._convert_gencode_alphabet()

        normalize_emission(aa_emission)
        self._aa_emission = aa_emission

        self._codon_emission = {}
        self._generate_codon_emission()
        normalize_emission(self._codon_emission)

    def _convert_gencode_alphabet(self):

        for aa in self._gencode.keys():
            self._gencode[aa] = [
                convert_to(codon, self._molecule) for codon in self._gencode[aa]
            ]

        return self._gencode

    @property
    def gencode(self):
        return self._gencode

    @property
    def amino_acids(self):
        return "".join(sorted(self._aa_emission.keys()))

    @property
    def bases(self):
        return self._molecule.bases

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
        from itertools import product

        for aa, nlogp in self._aa_emission.items():
            codons = self._gencode.get(aa, [])
            norm = log(len(codons))
            for codon in codons:
                self._codon_emission.update({codon: nlogp + norm})

        # for a, b, c in product(*[self.bases] * 3):
        #     codon = a + b + c
        #     if codon not in self._codon_emission:
        #         self._codon_emission[codon] = inf


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

