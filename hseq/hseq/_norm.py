def _normalize_nlogspace(values):
    from numpy import asarray
    from scipy.special import logsumexp

    values = asarray(values, float)
    return logsumexp(-values) + values


def normalize_emission(emission):
    keys = list(emission.keys())
    nlogp = [emission[a] for a in keys]
    nlogp = _normalize_nlogspace(nlogp)
    emission.update({a: logp for a, logp in zip(keys, nlogp)})
