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


# with open("emission.txt", "r") as fp:
#     aa_emission = parse_aa_emission(fp)

# bases = "ACGU"
# aa2codon = AA2Codon(bases, gencode, aa_emission)
# print(sum(v for v in aa2codon.aa_emission().values()))
# print(sum(v for v in aa2codon.codon_emission().values()))

# p = Prob(bases, aa2codon.codon_emission(True), 0.1)

# print(p)
# sys.exit(0)


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
