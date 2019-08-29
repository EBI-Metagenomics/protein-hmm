from io import StringIO


def parse_emission(stream: StringIO, nlog_space=True):
    from math import log

    emission = {}
    for line in stream:
        k, v = line.strip().split(" ", 2)
        emission[k] = float(v)

    if not nlog_space:
        emission = {k: -log(v) for k, v in emission.items()}

    return emission

