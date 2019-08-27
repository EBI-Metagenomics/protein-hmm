from matplotlib import pyplot as plt
import numpy as np
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


def prob(m, epsilon):
    """ Probability of `m` base indels. """
    e = epsilon
    if m not in [0, 1, 2, 3, 4]:
        raise ValueError()
    return binomial(4, 4 - m) * (1 - e) ** (4 - m) * e ** m


epsilon = np.linspace(0, 0.5)

fig = plt.figure(figsize=(4, 3))
plt.plot(epsilon, prob(0, epsilon), label="$p(M=0)$", color="black", ls="--")
plt.plot(epsilon, prob(1, epsilon), label="$p(M=1)$", color="black", ls="-.")
plt.plot(epsilon, prob(2, epsilon), label="$p(M=2)$", color="black", ls=":")
plt.plot(epsilon, prob(3, epsilon), label="$p(M=3)$", color="black", ls="-")
plt.xlabel("$\epsilon$")
plt.ylabel("probability")
plt.legend()
plt.tight_layout()
plt.savefig("indel-prob.pdf")
