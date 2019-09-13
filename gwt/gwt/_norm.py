def normalize_emission(emission: dict):
    keys = list(emission.keys())
    logprobs = _normalize_logspace([emission[a] for a in keys])
    emission.update({a: logp for a, logp in zip(keys, logprobs)})


def _normalize_logspace(values):
    from numpy import asarray
    from scipy.special import logsumexp

    values = asarray(values, float)
    return [v - logsumexp(values) for v in values]
