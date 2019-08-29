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

    def __str__(self):
        return f"<{self._name}>"


class NormalState(State):
    def __init__(self, name: str, end_state: bool, emission: dict):
        from ._norm import normalize_emission

        alphabet = "".join(list(emission.keys()))
        normalize_emission(emission)
        self._emission = emission

        super(NormalState, self).__init__(name, end_state, alphabet)

    def emit(self, random):
        from math import exp

        abc = list(self._alphabet)
        probs = [exp(-self._emission[a]) for a in abc]
        return random.choice(abc, p=probs)

    def __str__(self):
        return f"<{self._name}>"

