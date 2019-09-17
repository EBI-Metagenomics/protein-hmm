from itertools import product
from scipy.special import logsumexp
from math import exp

from ._log import LOG
from ._norm import normalize_emission
from ._state import State, emission_table

_ = None


class FrameState(State):
    def __init__(
        self, name: str, base_emission: dict, codon_emission: dict, epsilon: float
    ):
        normalize_emission(base_emission)
        self._base_emission = base_emission
        alphabet = "".join(base_emission.keys())

        normalize_emission(codon_emission)
        self._cemission = codon_emission

        self._epsilon = epsilon
        self._loge = LOG(epsilon)
        self._log1e = LOG(1 - epsilon)
        super(FrameState, self).__init__(name, alphabet, False)

    def prob(self, z, log_space: bool = False):
        """ p(Z=z1z2...zf, F=f). """
        f = len(z)
        p: float = 0.0
        if f == 1:
            p = self._joint_z_f1(*z)
        elif f == 2:
            p = self._joint_z_f2(*z)
        elif f == 3:
            p = self._joint_z_f3(*z)
        elif f == 4:
            p = self._joint_z_f4(*z)
        elif f == 5:
            p = self._joint_z_f5(*z)
        else:
            p = LOG(0.0)
        if log_space:
            return p
        return exp(p)

    def emit(self, random):
        lengths = [1, 2, 3, 4, 5]
        probs = [exp(self._len_prob(f)) for f in [1, 2, 3, 4, 5]]
        f = random.choice(lengths, p=probs)

        def prob(z):
            return exp(self._prob_z_given_f(z))

        abc = self._alphabet
        emission = {"".join(z): prob(z) for z in product(*[abc] * f)}
        seq = random.choice(list(emission.keys()), p=list(emission.values()))
        return seq

    def emission(self, log_space: bool = False):
        """
        Parameters
        ----------
        log_space : bool
            ``True`` to return in log space. Defaults to ``False``.
        """
        abc = self._alphabet
        emission = {}
        for f in range(1, 6):
            combs = product(*[abc] * f)
            emission.update({"".join(z): exp(self._joint_z_f(z)) for z in combs})
        emission = {a: LOG(b) for a, b in emission.items()}
        return emission_table(emission, log_space)

    @property
    def min_len(self):
        return 1

    @property
    def max_len(self):
        return 5

    def _codon_prob(self, x1, x2, x3):

        if x1 is None:
            x1 = self.alphabet
        if x2 is None:
            x2 = self.alphabet
        if x3 is None:
            x3 = self.alphabet

        get = self._cemission.get
        probs = [get(a + b + c, LOG(0.0)) for a, b, c in product(x1, x2, x3)]
        return logsumexp(probs)

    def _joint_z_f(self, z):
        if len(z) == 1:
            return self._joint_z_f1(*z)
        elif len(z) == 2:
            return self._joint_z_f2(*z)
        elif len(z) == 3:
            return self._joint_z_f3(*z)
        elif len(z) == 4:
            return self._joint_z_f4(*z)
        elif len(z) == 5:
            return self._joint_z_f5(*z)

    def _joint_z_f1(self, z1):
        """ p(Z=z1, F=1). """
        e = self._codon_prob
        c = 2 * self._loge + 2 * self._log1e
        return c + logsumexp([e(z1, _, _), e(_, z1, _), e(_, _, z1)]) - LOG(3)

    def _joint_z_f2(self, z1, z2):
        """ p(Z=z1z2, F=2). """
        i = self._base_emission.get
        e = self._codon_prob

        c = LOG(2) + self._loge + self._log1e * 3 - LOG(3)
        p = [c + logsumexp([e(_, z1, z2), e(z1, _, z2), e(z1, z2, _)])]

        c = 3 * self._loge + self._log1e - LOG(3)
        p += [c + logsumexp([e(z1, _, _), e(_, z1, _), e(_, _, z1)]) + i(z2)]
        p += [c + logsumexp([e(z2, _, _), e(_, z2, _), e(_, _, z2)]) + i(z1)]

        return logsumexp(p)

    def _joint_z_f3(self, z1, z2, z3):
        """ p(Z=z1z2z3, F=3). """
        i = self._base_emission.get
        e = self._codon_prob
        p0 = 4 * self._log1e + e(z1, z2, z3)

        c = LOG(4) + 2 * self._loge + 2 * self._log1e - LOG(9)
        p = [logsumexp([e(_, z2, z3), e(z2, _, z3), e(z2, z3, _)]) + i(z1)]
        p += [logsumexp([e(_, z1, z3), e(z1, _, z3), e(z1, z3, _)]) + i(z2)]
        p += [logsumexp([e(_, z1, z2), e(z1, _, z2), e(z1, z2, _)]) + i(z3)]
        p1 = c + logsumexp(p)

        c = 4 * self._loge - LOG(9)
        p = [logsumexp([e(z3, _, _), e(_, z3, _), e(_, _, z3)]) + i(z1) + i(z2)]
        p += [logsumexp([e(z2, _, _), e(_, z2, _), e(_, _, z2)]) + i(z1) + i(z3)]
        p += [logsumexp([e(z1, _, _), e(_, z1, _), e(_, _, z1)]) + i(z2) + i(z3)]
        p2 = c + logsumexp(p)

        return logsumexp([p0, p1, p2])

    def _joint_z_f4(self, z1, z2, z3, z4):
        """ p(Z=z1z2...z4, F=4). """
        i = self._base_emission.get
        e = self._codon_prob

        c = self._loge + self._log1e * 3 - LOG(2)
        p = [logsumexp([e(z2, z3, z4) + i(z1), e(z1, z3, z4) + i(z2)])]
        p += [logsumexp([e(z1, z2, z4) + i(z3), e(z1, z2, z3) + i(z4)])]
        p0 = c + logsumexp(p)

        c = 3 * self._loge + self._log1e - LOG(9)
        p = [logsumexp([e(_, z3, z4) + i(z1) + i(z2), e(_, z2, z4) + i(z1) + i(z3)])]
        p += [logsumexp([e(_, z2, z3) + i(z1) + i(z4), e(_, z1, z4) + i(z2) + i(z3)])]
        p += [logsumexp([e(_, z1, z3) + i(z2) + i(z4), e(_, z1, z2) + i(z3) + i(z4)])]
        p += [logsumexp([e(z3, _, z4) + i(z1) + i(z2), e(z2, _, z4) + i(z1) + i(z3)])]
        p += [logsumexp([e(z2, _, z3) + i(z1) + i(z4), e(z1, _, z4) + i(z2) + i(z3)])]
        p += [logsumexp([e(z1, _, z3) + i(z2) + i(z4), e(z1, _, z2) + i(z3) + i(z4)])]
        p += [logsumexp([e(z3, z4, _) + i(z1) + i(z2), e(z2, z4, _) + i(z1) + i(z3)])]
        p += [logsumexp([e(z2, z3, _) + i(z1) + i(z4), e(z1, z4, _) + i(z2) + i(z3)])]
        p += [logsumexp([e(z1, z3, _) + i(z2) + i(z4), e(z1, z2, _) + i(z3) + i(z4)])]
        p1 = c + logsumexp(p)

        return logsumexp([p0, p1])

    def _joint_z_f5(self, z1, z2, z3, z4, z5):
        """ p(Z=z1z2...z5, F=5). """
        i = self._base_emission.get
        e = self._codon_prob

        c = 2 * self._loge + 2 * self._log1e - LOG(10)
        p = [logsumexp([i(z1) + i(z2) + e(z3, z4, z5), i(z1) + i(z3) + e(z2, z4, z5)])]
        p += [logsumexp([i(z1) + i(z4) + e(z2, z3, z5), i(z1) + i(z5) + e(z2, z3, z4)])]
        p += [logsumexp([i(z2) + i(z3) + e(z1, z4, z5), i(z2) + i(z4) + e(z1, z3, z5)])]
        p += [logsumexp([i(z2) + i(z5) + e(z1, z3, z4), i(z3) + i(z4) + e(z1, z2, z5)])]
        p += [logsumexp([i(z3) + i(z5) + e(z1, z2, z4), i(z4) + i(z5) + e(z1, z2, z3)])]

        return c + logsumexp(p)

    def _prob_z_given_f(self, z):
        """ p(Z=z1z2...zf | F=f). """
        f = len(z)
        if f == 1:
            return self._joint_z_f1(*z) - self._len_prob(1)
        elif f == 2:
            return self._joint_z_f2(*z) - self._len_prob(2)
        elif f == 3:
            return self._joint_z_f3(*z) - self._len_prob(3)
        elif f == 4:
            return self._joint_z_f4(*z) - self._len_prob(4)
        elif f == 5:
            return self._joint_z_f5(*z) - self._len_prob(5)

    def _len_prob(self, f):
        """ P(F=f). """
        e = self._loge
        e1 = self._log1e
        if f == 1 or f == 5:
            return 2 * e + 2 * e1
        elif f == 2 or f == 4:
            return logsumexp([LOG(2) + 3 * e + e1, LOG(2) + e + 3 * e1])
        elif f == 3:
            return logsumexp([4 * e, LOG(4) + 2 * e + 2 * e1, 4 * e1])

    def __repr__(self):
        return f"<{self.__class__.__name__}:{self._name}>"
