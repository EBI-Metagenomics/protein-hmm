from itertools import product
from math import exp, factorial as fac

from ._molecule import Molecule
from ._norm import normalize_emission
from ._log import LOG


class FrameEmission:
    def __init__(
        self,
        codon_emission: dict,
        molecule: Molecule,
        epsilon: float,
        base_emission: dict = None,
    ):
        """
        Parameters
        ----------
        codon_emission : dict
            Codon emission probabilities in log space.
        molecule : Molecule
            RNA or DNA molecule.
        epsilon : float
            Transition probability of indel.
        """
        normalize_emission(codon_emission)
        self._molecule = molecule
        self._codon_emission = codon_emission
        self._epsilon = epsilon
        if base_emission is None:
            logp = -LOG(len(molecule.bases))
            base_emission = {base: logp for base in molecule.bases}
        normalize_emission(base_emission)
        self._base_emission = base_emission

    @property
    def bases(self):
        return self._molecule.bases

    def len_prob(self, f):
        """ P(F=f). """
        e = self._epsilon
        if f == 1 or f == 5:
            return e ** 2 * (1 - e) ** 2
        elif f == 2 or f == 4:
            return 2 * e ** 3 * (1 - e) + 2 * e * (1 - e) ** 3
        elif f == 3:
            return e ** 4 + 4 * e ** 2 * (1 - e) ** 2 + (1 - e) ** 4
        return 0.0

    def indel_prob(self, m):
        """ Probability of `m` base indels. """
        e = self._epsilon
        if m not in [0, 1, 2, 3, 4]:
            return 0.0
        return binomial(4, 4 - m) * (1 - e) ** (4 - m) * e ** m

    def prob(self, z):
        """ p(Z=z1z2...zf, F=f). """
        f = len(z)
        eps = self._epsilon
        _ = None
        i = self._base_bg_prob
        e = self._codon_prob
        p = 0
        if f == 1:
            z1 = z[0]

            c = eps ** 2 * (1 - eps) ** 2
            p = c * (e(z1, _, _) + e(_, z1, _) + e(_, _, z1)) / 3

        elif f == 2:
            z1 = z[0]
            z2 = z[1]

            c = 2 * eps * (1 - eps) ** 3 / 3
            p = c * (e(_, z1, z2) + e(z1, _, z2) + e(z1, z2, _))

            c = eps ** 3 * (1 - eps) / 3
            p += c * (e(z1, _, _) + e(_, z1, _) + e(_, _, z1)) * i(z2)
            p += c * (e(z2, _, _) + e(_, z2, _) + e(_, _, z2)) * i(z1)

        elif f == 3:
            z1 = z[0]
            z2 = z[1]
            z3 = z[2]

            p += (1 - eps) ** 4 * e(z1, z2, z3)

            c = 4 * eps ** 2 * (1 - eps) ** 2 / 9
            p += c * (e(_, z2, z3) + e(z2, _, z3) + e(z2, z3, _)) * i(z1)
            p += c * (e(_, z1, z3) + e(z1, _, z3) + e(z1, z3, _)) * i(z2)
            p += c * (e(_, z1, z2) + e(z1, _, z2) + e(z1, z2, _)) * i(z3)

            c = eps ** 4 / 9
            p += c * (e(z3, _, _) + e(_, z3, _) + e(_, _, z3)) * i(z1) * i(z2)
            p += c * (e(z2, _, _) + e(_, z2, _) + e(_, _, z2)) * i(z1) * i(z3)
            p += c * (e(z1, _, _) + e(_, z1, _) + e(_, _, z1)) * i(z2) * i(z3)

        elif f == 4:
            z1 = z[0]
            z2 = z[1]
            z3 = z[2]
            z4 = z[3]

            c = eps * (1 - eps) ** 3 / 2
            p += c * (e(z2, z3, z4) * i(z1) + e(z1, z3, z4) * i(z2))
            p += c * (e(z1, z2, z4) * i(z3) + e(z1, z2, z3) * i(z4))

            c = eps ** 3 * (1 - eps) / 9
            p += c * (e(_, z3, z4) * i(z1) * i(z2) + e(_, z2, z4) * i(z1) * i(z3))
            p += c * (e(_, z2, z3) * i(z1) * i(z4) + e(_, z1, z4) * i(z2) * i(z3))
            p += c * (e(_, z1, z3) * i(z2) * i(z4) + e(_, z1, z2) * i(z3) * i(z4))

            p += c * (e(z3, _, z4) * i(z1) * i(z2) + e(z2, _, z4) * i(z1) * i(z3))
            p += c * (e(z2, _, z3) * i(z1) * i(z4) + e(z1, _, z4) * i(z2) * i(z3))
            p += c * (e(z1, _, z3) * i(z2) * i(z4) + e(z1, _, z2) * i(z3) * i(z4))

            p += c * (e(z3, z4, _) * i(z1) * i(z2) + e(z2, z4, _) * i(z1) * i(z3))
            p += c * (e(z2, z3, _) * i(z1) * i(z4) + e(z1, z4, _) * i(z2) * i(z3))
            p += c * (e(z1, z3, _) * i(z2) * i(z4) + e(z1, z2, _) * i(z3) * i(z4))

        elif f == 5:
            z1 = z[0]
            z2 = z[1]
            z3 = z[2]
            z4 = z[3]
            z5 = z[4]

            c = eps ** 2 * (1 - eps) ** 2 / 10
            p += c * (i(z1) * i(z2) * e(z3, z4, z5) + i(z1) * i(z3) * e(z2, z4, z5))
            p += c * (i(z1) * i(z4) * e(z2, z3, z5) + i(z1) * i(z5) * e(z2, z3, z4))
            p += c * (i(z2) * i(z3) * e(z1, z4, z5) + i(z2) * i(z4) * e(z1, z3, z5))
            p += c * (i(z2) * i(z5) * e(z1, z3, z4) + i(z3) * i(z4) * e(z1, z2, z5))
            p += c * (i(z3) * i(z5) * e(z1, z2, z4) + i(z4) * i(z5) * e(z1, z2, z3))

        return p

    def emission(self):

        sequences = {}

        bases = list(self.bases)
        for f in range(1, 6):
            seqs = {"".join(seq): self.prob(seq) for seq in product(*[bases] * f)}
            sequences.update(seqs)

        items = sorted(sequences.items(), key=lambda x: x[1], reverse=True)
        return list(items)

    def _codon_prob(self, x1, x2, x3):
        from scipy.special import logsumexp

        p = []
        if all([x1 is None, x2 is None, x3 is not None]):
            for x1, x2 in product(self.bases, self.bases):
                p.append(self._codon_emission.get(x1 + x2 + x3, LOG(0.0)))

        elif all([x1 is None, x2 is not None, x3 is None]):
            for x1, x3 in product(self.bases, self.bases):
                p.append(self._codon_emission.get(x1 + x2 + x3, LOG(0.0)))

        elif all([x1 is not None, x2 is None, x3 is None]):
            for x2, x3 in product(self.bases, self.bases):
                p.append(self._codon_emission.get(x1 + x2 + x3, LOG(0.0)))

        elif all([x1 is not None, x2 is not None, x3 is None]):
            for x3 in self.bases:
                p.append(self._codon_emission.get(x1 + x2 + x3, LOG(0.0)))

        elif all([x1 is not None, x2 is None, x3 is not None]):
            for x2 in self.bases:
                p.append(self._codon_emission.get(x1 + x2 + x3, LOG(0.0)))

        elif all([x1 is None, x2 is not None, x3 is not None]):
            for x1 in self.bases:
                p.append(self._codon_emission.get(x1 + x2 + x3, LOG(0.0)))

        elif all([x1 is not None, x2 is not None, x3 is not None]):
            p.append(self._codon_emission.get(x1 + x2 + x3, LOG(0.0)))

        return exp(logsumexp(p))

    def _base_bg_prob(self, x):
        return exp(self._base_emission[x])

    def _get_codon_emission(self):
        return [(k, exp(v)) for (k, v) in self._codon_emission.items()]

    def __str__(self):
        msg = f"Epsilon = {self._epsilon}\n"

        msg += "\n"
        for m in range(0, 5):
            msg += f"p(M={m}) = {self.indel_prob(m):.4f}\n"

        msg += "\n"
        for f in range(1, 6):
            msg += f"p(F={f}) = {self.len_prob(f):.4f}\n"

        msg += "\n"
        msg += "Top codons\n"
        items = sorted(self._get_codon_emission(), key=lambda x: x[1], reverse=True)
        for c, v in items[:5]:
            msg += f"p(X={c}) = {v:.4f}\n"

        sequences = {}

        bases = self._molecule.bases
        for f in range(1, 6):
            msg += "\n"
            msg += f"Top sequences for F={f}\n"
            seqs = {"".join(seq): self.prob(seq) for seq in product(*[bases] * f)}
            sequences.update(seqs)
            items = sorted(seqs.items(), key=lambda x: x[1], reverse=True)
            norm = self.len_prob(f)
            for c, v in items[:5]:
                v /= norm
                msg += f"p(Z={c} | F={f}) = {v:.4f}\n"

        msg += "\n"
        msg += f"Top final sequences\n"
        items = sorted(sequences.items(), key=lambda x: x[1], reverse=True)
        for c, v in items[:100]:
            f = len(c)
            c += " " * (5 - len(c))
            msg += f"p(Z={c}, F={f}) = {v:.4f}\n"

        return msg


def binomial(n, k):
    """ Choose `k` from `n`. """
    return fac(n) // fac(k) // fac(n - k)
