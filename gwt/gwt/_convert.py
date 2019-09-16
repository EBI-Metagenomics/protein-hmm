from math import exp

from ._gencode import GENCODE
from ._molecule import Molecule, convert_to
from ._log import LOG
from ._molecule import RNA
from ._norm import normalize_emission
from hmmer_reader import HMMEReader
from hseq import HMM, SilentState, FrameState


class AA2Codon:
    def __init__(self, aa_emission: dict, molecule: Molecule, gencode="standard"):
        """
        Parameters
        ----------
        aa_emission : dict
            Amino acid emission probabilities in log space.
        molecule : Molecule
            Molecule.
        """
        self._gencode = GENCODE[gencode].copy()
        self._molecule = molecule
        _convert_gencode_alphabet(self._gencode, molecule)

        normalize_emission(aa_emission)
        self._aa_emission = aa_emission

        self._codon_emission = {}
        self._generate_codon_emission()
        normalize_emission(self._codon_emission)

    @property
    def gencode(self):
        return self._gencode

    @property
    def amino_acids(self):
        return "".join(sorted(self._aa_emission.keys()))

    @property
    def bases(self):
        return self._molecule.bases

    def aa_emission(self, log_space: bool = False):
        """
        Parameters
        ----------
        log_space : bool
            ``True`` to return the amino acid emission probabilities in log space.
            Defaults to ``False``.
        """
        if log_space:
            return self._aa_emission
        return {k: exp(v) for k, v in self._aa_emission.items()}

    def codon_emission(self, log_space: bool = False):
        """
        Parameters
        ----------
        log_space : bool
            ``True`` to return the codon emission probabilities in log space.
            Defaults to ``False``.
        """
        if log_space:
            return self._codon_emission
        return {k: exp(v) for k, v in self._codon_emission.items()}

    def _generate_codon_emission(self):
        for aa, logp in self._aa_emission.items():
            codons = self._gencode.get(aa, [])
            logp_norm = LOG(len(codons))
            for codon in codons:
                self._codon_emission.update({codon: logp - logp_norm})


def create_frame_hmm(
    hmmfile: HMMEReader, phmm: HMM, epsilon: float, gencode="standard"
):
    molecule = RNA()
    alphabet = molecule.bases
    hmm = HMM(alphabet)

    gcode = GENCODE[gencode].copy()
    _convert_gencode_alphabet(gcode, molecule)
    base_compo = _infer_base_compo(_infer_codon_compo(hmmfile.compo, gcode), molecule)

    hmm.add_state(SilentState("M0", alphabet, False), LOG(1.0))

    def aa2codon(state):
        convert = AA2Codon(dict(phmm.states[state].emission(True)), molecule)
        return convert.codon_emission(True)

    hmm.add_state(FrameState("I0", base_compo, aa2codon("I0"), epsilon))
    hmm.add_state(SilentState("D0", alphabet, False))

    for m in range(1, hmmfile.M + 1):
        hmm.add_state(FrameState(f"M{m}", base_compo, aa2codon(f"M{m}"), epsilon))
        hmm.add_state(FrameState(f"I{m}", base_compo, aa2codon(f"I{m}"), epsilon))
        hmm.add_state(SilentState(f"D{m}", alphabet, False))

        trans = hmmfile.trans(m - 1, True)

        hmm.set_trans(f"M{m-1}", f"M{m}", trans["MM"])
        hmm.set_trans(f"M{m-1}", f"I{m-1}", trans["MI"])
        hmm.set_trans(f"M{m-1}", f"D{m}", trans["MD"])
        hmm.set_trans(f"I{m-1}", f"M{m}", trans["IM"])
        hmm.set_trans(f"I{m-1}", f"I{m-1}", trans["II"])
        hmm.set_trans(f"D{m-1}", f"M{m}", trans["DM"])
        hmm.set_trans(f"D{m-1}", f"D{m}", trans["DD"])

    hmm.add_state(SilentState("E", alphabet, True))
    hmm.set_trans("E", "E", LOG(1.0))

    M = hmmfile.M
    trans = hmmfile.trans(M, True)
    hmm.set_trans(f"M{M}", f"E", trans["MM"])
    hmm.set_trans(f"M{M}", f"I{M}", trans["MI"])
    hmm.set_trans(f"I{M}", f"E", trans["IM"])
    hmm.set_trans(f"I{M}", f"I{M}", trans["II"])
    hmm.set_trans(f"D{M}", f"E", trans["DM"])

    hmm.rename_state("M0", "B")
    hmm.delete_state("D0")
    hmm.normalize()
    return hmm


def create_bg_frame_hmm(
    hmmfile: HMMEReader, phmm: HMM, epsilon: float, gencode="standard"
):
    molecule = RNA()
    alphabet = molecule.bases
    hmm = HMM(alphabet)

    gcode = GENCODE[gencode].copy()
    _convert_gencode_alphabet(gcode, molecule)
    base_compo = _infer_base_compo(_infer_codon_compo(hmmfile.compo, gcode), molecule)

    def aa2codon(state):
        convert = AA2Codon(dict(phmm.states[state].emission(True)), molecule)
        return convert.codon_emission(True)

    hmm.add_state(FrameState("I", base_compo, aa2codon("I1"), epsilon), LOG(1.0))
    hmm.normalize()
    return hmm


def _infer_codon_compo(aa_compo, gencode):
    codon_compo = []
    for aa, logp in aa_compo.items():
        logp_norm = LOG(len(gencode[aa]))
        for codon in gencode[aa]:
            codon_compo.append((codon, logp - logp_norm))
    return codon_compo


def _infer_base_compo(codon_compo, molecule: Molecule):
    from ._norm import normalize_emission
    from numpy import asarray
    from scipy.special import logsumexp

    base_compo = {base: [] for base in molecule.bases}
    logp_norm = LOG(3)
    for codon, logp in codon_compo:
        base_compo[codon[0]] += [logp - logp_norm]
        base_compo[codon[1]] += [logp - logp_norm]
        base_compo[codon[2]] += [logp - logp_norm]

    for base in base_compo.keys():
        base_compo[base] = logsumexp(asarray(base_compo[base], float))

    normalize_emission(base_compo)
    return base_compo


def _convert_gencode_alphabet(gencode, molecule: Molecule):
    for aa in gencode.keys():
        gencode[aa] = [convert_to(c, molecule) for c in gencode[aa]]
