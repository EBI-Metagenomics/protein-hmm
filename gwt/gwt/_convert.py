from math import exp, log

from ._gencode import GENCODE
from ._molecule import Molecule, convert_to
from ._nlog import nlog
from ._molecule import RNA
from ._norm import normalize_emission
from hmmer_reader import HMMEReader
from hseq import HMM, SilentState, FrameState


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


def create_frame_hmm(hmmfile: HMMEReader, phmm: HMM, epsilon: float):
    molecule = RNA()
    alphabet = molecule.bases
    base_emission = {a: nlog(1.0 / len(alphabet)) for a in alphabet}
    hmm = HMM(alphabet)

    start_state = SilentState("S", alphabet, False)
    hmm.add_state(start_state, nlog(1.0))

    mat_state = SilentState("M0", alphabet, False)
    hmm.add_state(mat_state)
    aa_emission = {i[0]: i[1] for i in phmm.states["I0"].emission(nlog_space=True)}
    aa2codon = AA2Codon(aa_emission, molecule)
    codon_emission = aa2codon.codon_emission(nlog_space=True)
    ins_state = FrameState("I0", base_emission, codon_emission, epsilon)
    hmm.add_state(ins_state)
    del_state = SilentState("D0", alphabet, False)
    hmm.add_state(del_state)
    hmm.set_trans("S", "M0", nlog(1.0))

    for m in range(1, hmmfile.M + 1):
        aa_emission = phmm.states[f"M{m}"].emission(nlog_space=True)
        aa_emission = {i[0]: i[1] for i in aa_emission}
        aa2codon = AA2Codon(aa_emission, molecule)
        codon_emission = aa2codon.codon_emission(nlog_space=True)
        mat_state = FrameState(f"M{m}", base_emission, codon_emission, epsilon)
        hmm.add_state(mat_state)

        aa_emission = phmm.states[f"I{m}"].emission(nlog_space=True)
        aa_emission = {i[0]: i[1] for i in aa_emission}
        aa2codon = AA2Codon(aa_emission, molecule)
        codon_emission = aa2codon.codon_emission(nlog_space=True)
        ins_state = FrameState(f"I{m}", base_emission, codon_emission, epsilon)
        hmm.add_state(ins_state)

        del_state = SilentState(f"D{m}", alphabet, False)
        hmm.add_state(del_state)

        trans = hmmfile.trans(m - 1, False)

        hmm.set_trans(f"M{m-1}", f"M{m}", trans["MM"])
        hmm.set_trans(f"M{m-1}", f"I{m-1}", trans["MI"])
        hmm.set_trans(f"M{m-1}", f"D{m}", trans["MD"])
        hmm.set_trans(f"I{m-1}", f"M{m}", trans["IM"])
        hmm.set_trans(f"I{m-1}", f"I{m-1}", trans["II"])
        hmm.set_trans(f"D{m-1}", f"M{m}", trans["DM"])
        hmm.set_trans(f"D{m-1}", f"D{m}", trans["DD"])

    end_state = SilentState("E", alphabet, True)
    hmm.add_state(end_state)
    hmm.set_trans("E", "E", nlog(1.0))

    m = hmmfile.M
    trans = hmmfile.trans(m, False)
    hmm.set_trans(f"M{m}", f"E", trans["MM"])
    hmm.set_trans(f"M{m}", f"I{m}", trans["MI"])
    hmm.set_trans(f"I{m}", f"E", trans["IM"])
    hmm.set_trans(f"I{m}", f"I{m}", trans["II"])
    hmm.set_trans(f"D{m}", f"E", trans["DM"])

    hmm.normalize()
    return hmm
