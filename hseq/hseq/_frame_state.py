from itertools import product
from ._state import State
from ._norm import normalize_emission
from math import exp, inf


class FrameState(State):
    def __init__(self, name: str, alphabet: list, codon_emission: dict, epsilon: float):

        self._epsilon = epsilon
        normalize_emission(codon_emission)
        self._cemission = codon_emission

        super(FrameState, self).__init__(name, False, alphabet)

    def _codon_prob(self, x1, x2, x3):
        from scipy.special import logsumexp

        if x1 is None:
            x1 = self.alphabet
        if x2 is None:
            x2 = self.alphabet
        if x3 is None:
            x3 = self.alphabet

        get = self._cemission.get
        probs = [-get(a + b + c, inf) for a, b, c in product(x1, x2, x3)]
        return exp(logsumexp(probs))

    # def emit(self, random):
    #     triplets = list(self._emission.keys())
    #     probs = [exp(-self._emission[t]) for t in triplets]
    #     return random.choice(triplets, p=probs)

    # def prob(self, seq, nlog_space=False):
    #     v = self._emission.get(seq, inf)
    #     if not nlog_space:
    #         v = exp(-v)
    #     return v
