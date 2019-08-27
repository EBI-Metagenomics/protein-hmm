from matplotlib import pyplot as plt
import numpy as np

plt.rc("font", family="serif")
plt.rc("xtick", labelsize="x-small")
plt.rc("ytick", labelsize="x-small")
plt.rc("text", usetex=True)


def prob(f, epsilon):
    e = epsilon
    if f == 1 or f == 5:
        return e ** 2 * (1 - e) ** 2
    elif f == 2 or f == 4:
        return 2 * e ** 3 * (1 - e) + 2 * e * (1 - e) ** 3
    elif f == 3:
        return e ** 4 + 4 * e ** 2 * (1 - e) ** 2 + (1 - e) ** 4
    raise ValueError()


epsilon = np.linspace(0, 0.5)

fig = plt.figure(figsize=(4, 3))
plt.plot(epsilon, prob(1, epsilon), label="$p(F=1)$", color="black", ls="-.")
plt.plot(epsilon, prob(2, epsilon), label="$p(F=2)$", color="black", ls=":")
plt.plot(epsilon, prob(3, epsilon), label="$p(F=3)$", color="black", ls="-")
plt.xlabel("$\epsilon$")
plt.ylabel("probability")
plt.legend()
plt.tight_layout()
plt.savefig("seq-len-prob.pdf")
