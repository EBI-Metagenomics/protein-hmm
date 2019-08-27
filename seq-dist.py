from matplotlib import pyplot as plt
from itertools import product
from numpy import linspace
from numpy.random import RandomState
from math import factorial as fac

plt.rc("font", family="serif")
plt.rc("xtick", labelsize="x-small")
plt.rc("ytick", labelsize="x-small")
plt.rc("text", usetex=True)


class Prob:
    def __init__(self, bases, codon_emission, epsilon):
        self._codon_emission = codon_emission
        self._bases = bases
        self._epsilon = epsilon

    def _codon_prob(self, x1, x2, x3):
        p = 0
        if all([x1 is None, x2 is None, x3 is not None]):
            for x1, x2 in product(self._bases, self._bases):
                p += self._codon_emission[x1 + x2 + x3]

        elif all([x1 is None, x2 is not None, x3 is None]):
            for x1, x3 in product(self._bases, self._bases):
                p += self._codon_emission[x1 + x2 + x3]

        elif all([x1 is not None, x2 is None, x3 is None]):
            for x2, x3 in product(self._bases, self._bases):
                p += self._codon_emission[x1 + x2 + x3]

        elif all([x1 is not None, x2 is not None, x3 is None]):
            for x3 in self._bases:
                p += self._codon_emission[x1 + x2 + x3]

        elif all([x1 is not None, x2 is None, x3 is not None]):
            for x2 in self._bases:
                p += self._codon_emission[x1 + x2 + x3]

        elif all([x1 is None, x2 is not None, x3 is not None]):
            for x1 in self._bases:
                p += self._codon_emission[x1 + x2 + x3]

        elif all([x1 is not None, x2 is not None, x3 is not None]):
            p = self._codon_emission[x1 + x2 + x3]

        else:
            raise ValueError()

        return p

    def _base_bg_prob(self, x):
        return 1.0 / 4

    def len_prob(self, f):
        e = self._epsilon
        if f == 1 or f == 5:
            return e ** 2 * (1 - e) ** 2
        elif f == 2 or f == 4:
            return 2 * e ** 3 * (1 - e) + 2 * e * (1 - e) ** 3
        elif f == 3:
            return e ** 4 + 4 * e ** 2 * (1 - e) ** 2 + (1 - e) ** 4
        raise ValueError()

    def prob(self, z):
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


def create_codon_emission(bases, random):
    random = RandomState(0)
    codon_emission = {
        x1 + x2 + x3: random.rand() for x1, x2, x3 in product(bases, bases, bases)
    }
    norm = sum(codon_emission.values())
    return {k: v / norm for k, v in codon_emission.items()}


bases = "ACGT"
random = RandomState(0)
codon_emission = create_codon_emission(bases, random)
p = Prob(bases, codon_emission, 0.1)


total = 0
for x1 in bases:
    t = p.prob(x1)
    total += t
    print("{}: {}".format(x1, t))
# print(total / p.len_prob(1))

# total = 0
for x1, x2 in product(*[bases] * 2):
    t = p.prob(x1 + x2)
    total += t
    print("{}{}: {}".format(x1, x2, t))
# print(total / p.len_prob(2))

# total = 0
for x1, x2, x3 in product(*[bases] * 3):
    t = p.prob(x1 + x2 + x3)
    total += t
    print("{}{}{}: {}".format(x1, x2, x3, t))
# print(total / p.len_prob(3))

# total = 0
for x1, x2, x3, x4 in product(*[bases] * 4):
    t = p.prob(x1 + x2 + x3 + x4)
    total += t
    print("{}{}{}{}: {}".format(x1, x2, x3, x4, t))
# print(total / p.len_prob(4))

# total = 0
for x1, x2, x3, x4, x5 in product(*[bases] * 5):
    t = p.prob(x1 + x2 + x3 + x4 + x5)
    total += t
    print("{}{}{}{}{}: {}".format(x1, x2, x3, x4, x5, t))
# print(total / p.len_prob(5))
print(total)
