from pathlib import Path
import hmmer_reader


def read_hmmer_file(filepath: Path):

    if not filepath.exists():
        raise ValueError(f"`{filepath}` does not exist.")

    if not filepath.is_file():
        raise ValueError(f"`{filepath}` is not a file.")

    return hmmer_reader.read(filepath)


def create_hmmer_profile(hmmfile: hmmer_reader.HMMEReader):
    import hseq
    from hseq import nlog

    alphabet = hmmfile.alphabet

    hmm = hseq.HMM(alphabet)

    hmm.add_state(hseq.SilentState("S", alphabet, False), nlog(1.0))
    hmm.add_state(hseq.SilentState("M0", alphabet, False))
    hmm.add_state(hseq.NormalState("I0", hmmfile.insert(0, False)))
    hmm.add_state(hseq.SilentState("D0", alphabet, False))
    hmm.set_trans("S", "M0", nlog(1.0))

    for m in range(1, hmmfile.M + 1):
        hmm.add_state(hseq.NormalState(f"M{m}", hmmfile.match(m, False)))
        hmm.add_state(hseq.NormalState(f"I{m}", hmmfile.insert(m, False)))
        hmm.add_state(hseq.SilentState(f"D{m}", alphabet, False))

        trans = hmmfile.trans(m - 1, False)

        hmm.set_trans(f"M{m-1}", f"M{m}", trans["MM"])
        hmm.set_trans(f"M{m-1}", f"I{m-1}", trans["MI"])
        hmm.set_trans(f"M{m-1}", f"D{m}", trans["MD"])
        hmm.set_trans(f"I{m-1}", f"M{m}", trans["IM"])
        hmm.set_trans(f"I{m-1}", f"I{m-1}", trans["II"])
        hmm.set_trans(f"D{m-1}", f"M{m}", trans["DM"])
        hmm.set_trans(f"D{m-1}", f"D{m}", trans["DD"])

    hmm.add_state(hseq.SilentState("E", alphabet, True))
    hmm.set_trans("E", "E", nlog(1.0))

    M = hmmfile.M
    trans = hmmfile.trans(M, False)
    hmm.set_trans(f"M{M}", f"E", trans["MM"])
    hmm.set_trans(f"M{M}", f"I{M}", trans["MI"])
    hmm.set_trans(f"I{M}", f"E", trans["IM"])
    hmm.set_trans(f"I{M}", f"I{M}", trans["II"])
    hmm.set_trans(f"D{M}", f"E", trans["DM"])

    hmm.normalize()
    return hmm
