from io import StringIO


def parse_emission(stream: StringIO):
    emission = {}
    for line in stream:
        k, v = line.strip().split(" ", 2)
        emission[k] = float(v)

    return emission
