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

    def _joint_z_f1(self, z1):
        """ p(Z=z1, F=1). """
        e = self._codon_prob
        c = self._epsilon ** 2 * (1 - self._epsilon) ** 2
        return c * (e(z1, _, _) + e(_, z1, _) + e(_, _, z1)) / 3

    def _joint_z_f2(self, z1, z2):
        """ p(Z=z1z2, F=2). """
        i = lambda x: exp(-self._base_emission.get(x))
        e = self._codon_prob

        c = 2 * self._epsilon * (1 - self._epsilon) ** 3 / 3
        p = c * (e(_, z1, z2) + e(z1, _, z2) + e(z1, z2, _))

        c = self._epsilon ** 3 * (1 - self._epsilon) / 3
        p += c * (e(z1, _, _) + e(_, z1, _) + e(_, _, z1)) * i(z2)
        p += c * (e(z2, _, _) + e(_, z2, _) + e(_, _, z2)) * i(z1)

        return p

    def _joint_z_f3(self, z1, z2, z3):
        """ p(Z=z1z2z3, F=3). """
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

    def _joint_z_f4(self, z1, z2, z3, z4):
        """ p(Z=z1z2...z4, F=4). """
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

    def _joint_z_f5(self, z1, z2, z3, z4, z5):
        """ p(Z=z1z2...z5, F=5). """
        i = lambda x: exp(-self._base_emission.get(x))
        e = self._codon_prob

        c = self._epsilon ** 2 * (1 - self._epsilon) ** 2 / 10
        p = c * (i(z1) * i(z2) * e(z3, z4, z5) + i(z1) * i(z3) * e(z2, z4, z5))
        p += c * (i(z1) * i(z4) * e(z2, z3, z5) + i(z1) * i(z5) * e(z2, z3, z4))
        p += c * (i(z2) * i(z3) * e(z1, z4, z5) + i(z2) * i(z4) * e(z1, z3, z5))
        p += c * (i(z2) * i(z5) * e(z1, z3, z4) + i(z3) * i(z4) * e(z1, z2, z5))
        p += c * (i(z3) * i(z5) * e(z1, z2, z4) + i(z4) * i(z5) * e(z1, z2, z3))

        return p

    def _prob_z_given_f(self, z):
        """ p(Z=z1z2...zf | F=f). """
        f = len(z)
        if f == 1:
            return self._joint_z_f1(*z) / self._len_prob(1)
        elif f == 2:
            return self._joint_z_f2(*z) / self._len_prob(2)
        elif f == 3:
            return self._joint_z_f3(*z) / self._len_prob(3)
        elif f == 4:
            return self._joint_z_f4(*z) / self._len_prob(4)
        elif f == 5:
            return self._joint_z_f5(*z) / self._len_prob(5)
        else:
            return 0.0

    def prob(self, z):
        """ p(Z=z1z2...zf, F=f). """
        f = len(z)
        if f == 1:
            return self._joint_z_f1(*z)
        elif f == 2:
            return self._joint_z_f2(*z)
        elif f == 3:
            return self._joint_z_f3(*z)
        elif f == 4:
            return self._joint_z_f4(*z)
        elif f == 5:
            return self._joint_z_f5(*z)
        else:
            return 0.0

    def _len_prob(self, f):
        """ P(F=f). """
        e = self._epsilon
        if f == 1 or f == 5:
            return e ** 2 * (1 - e) ** 2
        elif f == 2 or f == 4:
            return 2 * e ** 3 * (1 - e) + 2 * e * (1 - e) ** 3
        elif f == 3:
            return e ** 4 + 4 * e ** 2 * (1 - e) ** 2 + (1 - e) ** 4
        return 0.0

    def emit(self, random):
        lengths = [1, 2, 3, 4, 5]
        probs = [self._len_prob(f) for f in [1, 2, 3, 4, 5]]
        f = random.choice(lengths, p=probs)

        abc = self._alphabet
        emission = {"".join(z): self._prob_z_given_f(z) for z in product(*[abc] * f)}
        seq = random.choice(list(emission.keys()), p=list(emission.values()))
        return seq

    def __repr__(self):
        return f"<{self.__class__.__name__}:{self._name}>"
