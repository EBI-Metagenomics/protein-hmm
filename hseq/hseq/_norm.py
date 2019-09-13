def normalize_emission(emission: dict):
    """
    Normalize probabilities given in log space.
    """
    keys = list(emission.keys())
    logp = [emission[a] for a in keys]
    logp = _normalize_logspace(logp)
    emission.update({a: logp for a, logp in zip(keys, logp)})


def _normalize_logspace(values):
    from numpy import asarray
    from scipy.special import logsumexp

    values = asarray(values, float)
    return values - logsumexp(values)
