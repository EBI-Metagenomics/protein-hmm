from math import exp, log

from ._gencode import GENCODE
from ._molecule import Molecule, convert_to
from ._norm import normalize_emission


class AA2Codon:
    def __init__(self, aa_emission, molecule: Molecule, gencode="standard"):
        self._gencode = GENCODE[gencode].copy()
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

    def aa_emission(self, nlog_space=False):
        if nlog_space:
            return self._aa_emission
        return {k: exp(-v) for k, v in self._aa_emission.items()}

    def codon_emission(self, nlog_space=False):
        if nlog_space:
            return self._codon_emission
        return {k: exp(-v) for k, v in self._codon_emission.items()}

    def _generate_codon_emission(self):
        for aa, nlogp in self._aa_emission.items():
            codons = self._gencode.get(aa, [])
            norm = log(len(codons))
            for codon in codons:
                self._codon_emission.update({codon: nlogp + norm})
