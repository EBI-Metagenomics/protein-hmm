from ._norm import normalize_emission
from math import exp


class State:
    def __init__(self, name: str, end_state: bool, alphabet: list):
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


class SilentState(State):
    def __init__(self, name: str, end_state: bool):
        super(SilentState, self).__init__(name, end_state, "")

    def emit(self, random):
        del random
        return ""


class NormalState(State):
    def __init__(self, name: str, emission: dict):
        alphabet = "".join(list(emission.keys()))
        normalize_emission(emission)
        self._emission = emission

        super(NormalState, self).__init__(name, False, alphabet)

    def emit(self, random):
        abc = list(self._alphabet)
        probs = [exp(-self._emission[a]) for a in abc]
        return random.choice(abc, p=probs)


class TripletState(State):
    def __init__(self, name: str, alphabet: list, emission: dict):

        normalize_emission(emission)
        self._emission = emission

        super(TripletState, self).__init__(name, False, alphabet)

    def emit(self, random):
        triplets = list(self._emission.keys())
        probs = [exp(-self._emission[t]) for t in triplets]
        return random.choice(triplets, p=probs)

