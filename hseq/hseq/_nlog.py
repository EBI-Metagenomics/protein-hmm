def NLOG(probability: float):
    from math import inf, log

    if probability == 1.0:
        return 0.0
    if probability == 0.0:
        return inf

    return -log(probability)
