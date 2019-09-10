from ._norm import normalize_emission
from ._nlog import nlog
from math import exp, inf


class State:
    def __init__(self, name: str, end_state: bool, alphabet: str):
        self._name = name
        self._end_state = end_state
        self._alphabet = alphabet

    @property
    def name(self):
        return self._name

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def end_state(self):
        return self._end_state

    def __str__(self):
        return f"<{self._name}>"

    def __repr__(self):
        return f"<State:{self._name}>"


class SilentState(State):
    def __init__(self, name: str, alphabet: str, end_state: bool):
        super(SilentState, self).__init__(name, end_state, alphabet)

    def emit(self, random):
        del random
        return ""

    def prob(self, seq: str, nlog_space=False):
        if seq == "":
            v = nlog(1.0)
        else:
            v = nlog(0.0)
        if not nlog_space:
            v = exp(-v)
        return v

    def emission(self, nlog_space=False):
        if nlog_space:
            return [("", nlog(1.0))]
        return [("", nlog(1.0))]

    @property
    def min_len(self):
        return 0

    @property
    def max_len(self):
        return 0

    def __repr__(self):
        return f"<{self.__class__.__name__}:{self._name}>"


class NormalState(State):
    def __init__(self, name: str, emission: dict):
        alphabet = "".join(list(emission.keys()))
        normalize_emission(emission)
        self._emission = emission

        super(NormalState, self).__init__(name, False, alphabet)

    def emit(self, random):
        probs = [exp(-self._emission[a]) for a in self._alphabet]
        return random.choice(list(self._alphabet), p=probs)

    def prob(self, seq: str, nlog_space=False):
        v = self._emission.get(seq, nlog(0.0))
        if not nlog_space:
            v = exp(-v)
        return v

    def emission(self, nlog_space=False):
        table = list(self._emission.items())
        table = sorted(table, key=lambda x: -x[1])
        if not nlog_space:
            table = [(row[0], exp(-row[1])) for row in table]
        return table

    @property
    def min_len(self):
        return 1

    @property
    def max_len(self):
        return 1

    def __repr__(self):
        return f"<{self.__class__.__name__}:{self._name}>"


class TripletState(State):
    def __init__(self, name: str, alphabet: str, emission: dict):

        normalize_emission(emission)
        self._emission = emission

        super(TripletState, self).__init__(name, False, alphabet)

    def emit(self, random):
        triplets = list(self._emission.keys())
        probs = [exp(-v) for v in self._emission.values()]
        return random.choice(triplets, p=probs)

    def prob(self, seq: str, nlog_space=False):
        v = self._emission.get(seq, nlog(0.0))
        if not nlog_space:
            v = exp(-v)
        return v

    def emission(self, nlog_space=False):
        table = list(self._emission.items())
        table = sorted(table, key=lambda x: -x[1])
        if not nlog_space:
            table = [(row[0], exp(-row[1])) for row in table]
        return table

    @property
    def min_len(self):
        return 3

    @property
    def max_len(self):
        return 3

    def __repr__(self):
        return f"<{self.__class__.__name__}:{self._name}>"


