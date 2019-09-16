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


def create_bg_hmmer_profile(hmmfile: hmmer_reader.HMMEReader):
    import hseq

    hmm = hseq.HMM(hmmfile.alphabet)
    hmm.add_state(hseq.NormalState("I", hmmfile.insert(1, True)), LOG(1.0))
    hmm.normalize()
    return hmm


def estimate_gumbel_r_params(hmm, bg_hmm, nsamples=1000):
    from tqdm import tqdm
    from math import floor
    from numpy.random import RandomState

    seqs = [hmm.emit(RandomState(seed)) for seed in range(10)]
    assert seqs[0][0][0] == "B"
    assert seqs[0][-1][0] == "E"

    lengths = 0
    for seq in seqs:
        lengths += sum([1 for v in seqs[0] if v[0].startswith("M")])
    length = floor(lengths / len(seqs))

    max_logps = []
    bg_max_logps = []
    for seed in tqdm(range(nsamples)):
        random = RandomState(seed)
        path = bg_hmm.emit(random, max_nstates=length)
        seq = "".join(p[1] for p in path)
        max_logps += [hmm.viterbi(seq, True)[0]]
        bg_max_logps += [bg_hmm.viterbi(seq, True)[0]]

    scores = [a - b for (a, b) in zip(max_logps, bg_max_logps)]
    loc, scale = _find_gumbel_params(scores)
    return loc, scale


def gumbel_r_pvalue(seq, hmm, bg_hmm, loc, scale):
    import scipy.stats as st

    V = hmm.viterbi(seq, True)[0] - bg_hmm.viterbi(seq, True)[0]
    return 1 - st.gumbel_r(loc=loc, scale=scale).cdf(V)


def _find_gumbel_params(scores):
    import scipy.stats as st

    params = st.gumbel_r.fit(scores)
    loc = params[-2]
    scale = params[-1]

    return loc, scale
