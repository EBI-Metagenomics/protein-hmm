import sys
from math import exp, log, inf
from matplotlib import pyplot as plt
from itertools import product
from numpy import linspace
from numpy.random import RandomState
from math import factorial as fac

plt.rc("font", family="serif")
plt.rc("xtick", labelsize="x-small")
plt.rc("ytick", labelsize="x-small")
plt.rc("text", usetex=True)


def binomial(n, k):
    """ Choose `k` from `n`. """
    try:
        binom = fac(n) // fac(k) // fac(n - k)
    except ValueError:
        binom = 0
    return binom


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

    def indel_prob(self, m):
        """ Probability of `m` base indels. """
        e = self._epsilon
        if m not in [0, 1, 2, 3, 4]:
            raise ValueError()
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

    def __str__(self):
        msg = f"Epsilon = {self._epsilon}\n"

        msg += "\n"
        for m in range(0, 4):
            msg += f"p(M={m}) = {self.indel_prob(m):.4f}\n"

        msg += "\n"
        for f in range(1, 6):
            msg += f"p(F={f}) = {self.len_prob(f):.4f}\n"

        msg += "\n"
        msg += "Top codons\n"
        items = sorted(self._codon_emission.items(), key=lambda x: x[1], reverse=True)
        for c, v in items[:5]:
            msg += f"p(X={c}) = {v:.4f}\n"

        sequences = {}

        for f in range(1, 6):
            msg += "\n"
            msg += f"Top sequences for F={f}\n"
            seqs = {"".join(seq): self.prob(seq) for seq in product(*[self._bases] * f)}
            sequences.update(seqs)
            items = sorted(seqs.items(), key=lambda x: x[1], reverse=True)
            norm = self.len_prob(f)
            for c, v in items[:5]:
                v /= norm
                msg += f"p(Z={c} | F={f}) = {v:.4f}\n"

        msg += "\n"
        msg += f"Top final sequences\n"
        items = sorted(sequences.items(), key=lambda x: x[1], reverse=True)
        for c, v in items[:30]:
            f = len(c)
            c += " " * (5 - len(c))
            msg += f"p(Z={c}, F={f}) = {v:.4f}\n"

        return msg


def create_codon_emission(bases, random):
    random = RandomState(0)
    codon_emission = {
        x1 + x2 + x3: random.rand() for x1, x2, x3 in product(bases, bases, bases)
    }
    norm = sum(codon_emission.values())
    return {k: v / norm for k, v in codon_emission.items()}


def normalize(emission):
    norm = sum(emission.values())
    return {k: v / norm for k, v in emission.items()}


gencode = {
    "F": ["UUU", "UUC"],
    "L": ["UUA", "UUG", "CUU", "CUC", "CUA", "CUG"],
    "I": ["AUU", "AUC", "AUA"],
    "M": ["AUG"],
    "V": ["GUU", "GUC", "GUA", "GUG"],
    "S": ["UCU", "UCC", "UCA", "UCG"],
    "P": ["CCU", "CCC", "CCA", "CCG"],
    "T": ["ACU", "ACC", "ACA", "ACG"],
    "A": ["GCU", "GCC", "GCA", "GCG"],
    "Y": ["UAU", "UAC"],
    "*": ["UAA", "UAG", "UGA"],
    "H": ["CAU", "CAC"],
    "Q": ["CAA", "CAG"],
    "N": ["AAU", "AAC"],
    "K": ["AAA", "AAG"],
    "D": ["GAU", "GAC"],
    "E": ["GAA", "GAG"],
    "C": ["UGU", "UGC"],
    "W": ["UGG"],
    "R": ["CGU", "CGC", "CGA", "CGG"],
    "S": ["AGU", "AGC"],
    "R": ["AGA", "AGG"],
    "G": ["GGU", "GGC", "GGA", "GGG"],
}


def normalize_nlogspace(values):
    from numpy import asarray
    from scipy.special import logsumexp

    values = asarray(values, float)
    norm = logsumexp(-values)
    return [norm + v for v in values]


def parse_aa_emission(lines):

    aa_emission = {}
    for line in lines:
        aa, logp = line.split()
        logp = float(logp)
        aa_emission.update({aa: logp})

    return aa_emission


def convert_aa_to_codon_emission(aa_emission):
    codon_emission = {}
    for aa, nlogp in aa_emission.items():
        codons = gencode.get(aa, [])
        norm = log(len(codons))
        for codon in codons:
            codon_emission.update({codon: nlogp + norm})

    return codon_emission


def fill_remaining_codon_emission(bases, codon_emission):
    from math import inf

    for a, b, c in product(*[bases] * 3):
        codon = a + b + c
        if codon not in codon_emission:
            codon_emission[codon] = inf

    return codon_emission


def normalize_emission(emission):
    keys = list(emission.keys())
    nlogp = [emission[a] for a in keys]
    nlogp = normalize_nlogspace(nlogp)
    emission.update({a: logp for a, logp in zip(keys, nlogp)})


class AA2Codon:
    def __init__(self, bases, gencode, aa_emission):
        self._bases = bases
        self._gencode = gencode

        normalize_emission(aa_emission)
        self._aa_emission = aa_emission

        self._codon_emission = {}
        self._generate_codon_emission()
        normalize_emission(self._codon_emission)

    @property
    def amino_acids(self):
        return list(sorted(self._aa_emission.keys()))

    def aa_emission(self, prob_space=True):
        if prob_space:
            f = lambda x: exp(-x)
        else:
            f = lambda x: x
        return {k: f(v) for k, v in self._aa_emission.items()}

    def codon_emission(self, prob_space=True):
        if prob_space:
            f = lambda x: exp(-x)
        else:
            f = lambda x: x
        return {k: f(v) for k, v in self._codon_emission.items()}

    def _generate_codon_emission(self):
        for aa, nlogp in self._aa_emission.items():
            codons = self._gencode.get(aa, [])
            norm = log(len(codons))
            for codon in codons:
                self._codon_emission.update({codon: nlogp + norm})

        for a, b, c in product(*[self._bases] * 3):
            codon = a + b + c
            if codon not in self._codon_emission:
                self._codon_emission[codon] = inf


with open("emission.txt", "r") as fp:
    aa_emission = parse_aa_emission(fp)

bases = "ACGU"
aa2codon = AA2Codon(bases, gencode, aa_emission)
print(sum(v for v in aa2codon.aa_emission().values()))
print(sum(v for v in aa2codon.codon_emission().values()))

p = Prob(bases, aa2codon.codon_emission(True), 0.1)

print(p)
sys.exit(0)


# total = 0
# for x1 in bases:
#     t = p.prob(x1)
#     total += t
#     print("{}: {}".format(x1, t))
# # print(total / p.len_prob(1))

# # total = 0
# for x1, x2 in product(*[bases] * 2):
#     t = p.prob(x1 + x2)
#     total += t
#     print("{}{}: {}".format(x1, x2, t))
# # print(total / p.len_prob(2))

# # total = 0
# for x1, x2, x3 in product(*[bases] * 3):
#     t = p.prob(x1 + x2 + x3)
#     total += t
#     print("{}{}{}: {}".format(x1, x2, x3, t))
# # print(total / p.len_prob(3))

# # total = 0
# for x1, x2, x3, x4 in product(*[bases] * 4):
#     t = p.prob(x1 + x2 + x3 + x4)
#     total += t
#     print("{}{}{}{}: {}".format(x1, x2, x3, x4, t))
# # print(total / p.len_prob(4))

# # total = 0
# for x1, x2, x3, x4, x5 in product(*[bases] * 5):
#     t = p.prob(x1 + x2 + x3 + x4 + x5)
#     total += t
#     print("{}{}{}{}{}: {}".format(x1, x2, x3, x4, x5, t))
# # print(total / p.len_prob(5))
# print(total)
