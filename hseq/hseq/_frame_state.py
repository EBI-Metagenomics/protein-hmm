from itertools import product
from ._state import State
from ._norm import normalize_emission
from math import exp, inf

_ = None


class FrameState(State):
    def __init__(
        self, name: str, base_emission: dict, codon_emission: dict, epsilon: float
    ):

        normalize_emission(base_emission)
        self._base_emission = base_emission

        normalize_emission(codon_emission)
        self._cemission = codon_emission

        self._epsilon = epsilon

        alphabet = "".join(base_emission.keys())
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

    def _prob_f1(self, z1):
        e = self._codon_prob
        c = self._epsilon ** 2 * (1 - self._epsilon) ** 2
        return c * (e(z1, _, _) + e(_, z1, _) + e(_, _, z1)) / 3

    def _prob_f2(self, z1, z2):
        i = lambda x: exp(-self._base_emission.get(x))
        e = self._codon_prob

        c = 2 * self._epsilon * (1 - self._epsilon) ** 3 / 3
        p = c * (e(_, z1, z2) + e(z1, _, z2) + e(z1, z2, _))

        c = self._epsilon ** 3 * (1 - self._epsilon) / 3
        p += c * (e(z1, _, _) + e(_, z1, _) + e(_, _, z1)) * i(z2)
        p += c * (e(z2, _, _) + e(_, z2, _) + e(_, _, z2)) * i(z1)

        return p

    def _prob_f3(self, z1, z2, z3):
        i = lambda x: exp(-self._base_emission.get(x))
        e = self._codon_prob

        p = (1 - self._epsilon) ** 4 * e(z1, z2, z3)

        c = 4 * self._epsilon ** 2 * (1 - self._epsilon) ** 2 / 9
        p += c * (e(_, z2, z3) + e(z2, _, z3) + e(z2, z3, _)) * i(z1)
        p += c * (e(_, z1, z3) + e(z1, _, z3) + e(z1, z3, _)) * i(z2)
        p += c * (e(_, z1, z2) + e(z1, _, z2) + e(z1, z2, _)) * i(z3)

        c = self._epsilon ** 4 / 9
        p += c * (e(z3, _, _) + e(_, z3, _) + e(_, _, z3)) * i(z1) * i(z2)
        p += c * (e(z2, _, _) + e(_, z2, _) + e(_, _, z2)) * i(z1) * i(z3)
        p += c * (e(z1, _, _) + e(_, z1, _) + e(_, _, z1)) * i(z2) * i(z3)

        return p

    def _prob_f4(self, z1, z2, z3, z4):
        i = lambda x: exp(-self._base_emission.get(x))
        e = self._codon_prob

        c = self._epsilon * (1 - self._epsilon) ** 3 / 2
        p = c * (e(z2, z3, z4) * i(z1) + e(z1, z3, z4) * i(z2))
        p += c * (e(z1, z2, z4) * i(z3) + e(z1, z2, z3) * i(z4))

        c = self._epsilon ** 3 * (1 - self._epsilon) / 9
        p += c * (e(_, z3, z4) * i(z1) * i(z2) + e(_, z2, z4) * i(z1) * i(z3))
        p += c * (e(_, z2, z3) * i(z1) * i(z4) + e(_, z1, z4) * i(z2) * i(z3))
        p += c * (e(_, z1, z3) * i(z2) * i(z4) + e(_, z1, z2) * i(z3) * i(z4))

        p += c * (e(z3, _, z4) * i(z1) * i(z2) + e(z2, _, z4) * i(z1) * i(z3))
        p += c * (e(z2, _, z3) * i(z1) * i(z4) + e(z1, _, z4) * i(z2) * i(z3))
        p += c * (e(z1, _, z3) * i(z2) * i(z4) + e(z1, _, z2) * i(z3) * i(z4))

        p += c * (e(z3, z4, _) * i(z1) * i(z2) + e(z2, z4, _) * i(z1) * i(z3))
        p += c * (e(z2, z3, _) * i(z1) * i(z4) + e(z1, z4, _) * i(z2) * i(z3))
        p += c * (e(z1, z3, _) * i(z2) * i(z4) + e(z1, z2, _) * i(z3) * i(z4))

        return p

    def _prob_f5(self, z1, z2, z3, z4, z5):
        i = lambda x: exp(-self._base_emission.get(x))
        e = self._codon_prob

        c = self._epsilon ** 2 * (1 - self._epsilon) ** 2 / 10
        p = c * (i(z1) * i(z2) * e(z3, z4, z5) + i(z1) * i(z3) * e(z2, z4, z5))
        p += c * (i(z1) * i(z4) * e(z2, z3, z5) + i(z1) * i(z5) * e(z2, z3, z4))
        p += c * (i(z2) * i(z3) * e(z1, z4, z5) + i(z2) * i(z4) * e(z1, z3, z5))
        p += c * (i(z2) * i(z5) * e(z1, z3, z4) + i(z3) * i(z4) * e(z1, z2, z5))
        p += c * (i(z3) * i(z5) * e(z1, z2, z4) + i(z4) * i(z5) * e(z1, z2, z3))

        return p

    def prob(self, z):
        """ p(Z=z1z2...zf, F=f). """
        f = len(z)
        if f == 1:
            return self._prob_f1(*z)
        elif f == 2:
            return self._prob_f2(*z)
        elif f == 3:
            return self._prob_f3(*z)
        elif f == 4:
            return self._prob_f4(*z)
        elif f == 5:
            return self._prob_f5(*z)
        else:
            return 0.0

    # def emit(self, random):
    #     triplets = list(self._emission.keys())
    #     probs = [exp(-self._emission[t]) for t in triplets]
    #     return random.choice(triplets, p=probs)

    # def prob(self, seq, nlog_space=False):
    #     v = self._emission.get(seq, inf)
    #     if not nlog_space:
    #         v = exp(-v)
    #     return v