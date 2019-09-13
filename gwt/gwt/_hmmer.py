from pathlib import Path

import hmmer_reader

from ._log import LOG


def read_hmmer_file(filepath: Path):

    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    if not filepath.exists():
        raise ValueError(f"`{filepath}` does not exist.")

    if not filepath.is_file():
        raise ValueError(f"`{filepath}` is not a file.")

    return hmmer_reader.read(filepath)


def create_hmmer_profile(hmmfile: hmmer_reader.HMMEReader):
    import hseq

    alphabet = hmmfile.alphabet

    hmm = hseq.HMM(alphabet)

    hmm.add_state(hseq.SilentState("M0", alphabet, False), LOG(1.0))
    hmm.add_state(hseq.NormalState("I0", hmmfile.insert(0, True)))
    hmm.add_state(hseq.SilentState("D0", alphabet, False))

    for m in range(1, hmmfile.M + 1):
        hmm.add_state(hseq.NormalState(f"M{m}", hmmfile.match(m, True)))
        hmm.add_state(hseq.NormalState(f"I{m}", hmmfile.insert(m, True)))
        hmm.add_state(hseq.SilentState(f"D{m}", alphabet, False))

        trans = hmmfile.trans(m - 1, True)

        hmm.set_trans(f"M{m-1}", f"M{m}", trans["MM"])
        hmm.set_trans(f"M{m-1}", f"I{m-1}", trans["MI"])
        hmm.set_trans(f"M{m-1}", f"D{m}", trans["MD"])
        hmm.set_trans(f"I{m-1}", f"M{m}", trans["IM"])
        hmm.set_trans(f"I{m-1}", f"I{m-1}", trans["II"])
        hmm.set_trans(f"D{m-1}", f"M{m}", trans["DM"])
        hmm.set_trans(f"D{m-1}", f"D{m}", trans["DD"])

    hmm.add_state(hseq.SilentState("E", alphabet, True))
    hmm.set_trans("E", "E", LOG(1.0))

    M = hmmfile.M
    trans = hmmfile.trans(M, True)
    hmm.set_trans(f"M{M}", f"E", trans["MM"])
    hmm.set_trans(f"M{M}", f"I{M}", trans["MI"])
    hmm.set_trans(f"I{M}", f"E", trans["IM"])
    hmm.set_trans(f"I{M}", f"I{M}", trans["II"])
    hmm.set_trans(f"D{M}", f"E", trans["DM"])

    hmm.delete_state("D0")
    hmm.rename_state("M0", "B")
    hmm.normalize()
    return hmm
