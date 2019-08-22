def prob(f, epsilon):
    e = epsilon
    if f == 1 or f == 5:
        return e ** 2 * (1 - e) ** 2
    elif f == 2 or f == 4:
        return 2 * e ** 3 * (1 - e) + 2 * e * (1 - e) ** 3
    elif f == 3:
        return e ** 4 + 4 * e ** 2 * (1 - e) ** 2 + (1 - e) ** 4
    raise ValueError()


if __name__ == "__main__":
    for epsilon in [1e-3, 1e-2, 1e-1]:
        print(f"For 𝜀 = {epsilon}")
        for f in range(1, 6):
            print("  p(F={}) = {}".format(f, prob(f, epsilon)))
        print("  p(F=any) = {}".format(sum(prob(f, epsilon) for f in range(1, 6))))

