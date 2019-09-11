from pathlib import Path


def read_hmmer(filepath: Path):
    import hmmer_reader
    import hseq
    from hseq import nlog

    if not filepath.exists():
        raise ValueError(f"`{filepath}` does not exist.")

    if not filepath.is_file():
        raise ValueError(f"`{filepath}` is not a file.")

    hmmfile = hmmer_reader.read(filepath)
    alphabet = hmmfile.alphabet

    hmm = hseq.HMM(alphabet)

    start_state = hseq.SilentState("S", alphabet, False)
    hmm.add_state(start_state, nlog(1.0))

    # mat_state = hseq.NormalState("M0", hmmfile.match(0, False))
    mat_state = hseq.SilentState("M0", alphabet, False)
    hmm.add_state(mat_state)
    ins_state = hseq.NormalState("I0", hmmfile.insert(0, False))
    hmm.add_state(ins_state)
    del_state = hseq.SilentState("D0", alphabet, False)
    hmm.add_state(del_state)
    hmm.set_trans("S", "M0", nlog(1.0))

    for m in range(1, hmmfile.M + 1):
        mat_state = hseq.NormalState(f"M{m}", hmmfile.match(m, False))
        hmm.add_state(mat_state)
        ins_state = hseq.NormalState(f"I{m}", hmmfile.insert(m, False))
        hmm.add_state(ins_state)
        del_state = hseq.SilentState(f"D{m}", alphabet, False)
        hmm.add_state(del_state)

        trans = hmmfile.trans(m - 1, False)

        hmm.set_trans(f"M{m-1}", f"M{m}", trans["MM"])
        hmm.set_trans(f"M{m-1}", f"I{m-1}", trans["MI"])
        hmm.set_trans(f"M{m-1}", f"D{m}", trans["MD"])
        hmm.set_trans(f"I{m-1}", f"M{m}", trans["IM"])
        hmm.set_trans(f"I{m-1}", f"I{m-1}", trans["II"])
        hmm.set_trans(f"D{m-1}", f"M{m}", trans["DM"])
        hmm.set_trans(f"D{m-1}", f"D{m}", trans["DD"])

    end_state = hseq.SilentState("E", alphabet, True)
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
