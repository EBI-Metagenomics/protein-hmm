from ._norm import normalize_emission
from ._log import LOG
from math import exp


class State:
    def __init__(self, name: str, alphabet: str, end_state: bool):
        """
        Parameters
        ----------
        name : str
            Name.
        alphabet : str
            Alphabet.
        end_state : bool
            End state.
        """
        self._name = name
        self._end_state = end_state
        self._alphabet = alphabet

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, v: str):
        self._name = v

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def end_state(self):
        return self._end_state

    def __str__(self):
        return f"<{self._name}>"


class SilentState(State):
    def __init__(self, name: str, alphabet: str, end_state: bool):
        """
        Parameters
        ----------
        name : str
            Name.
        alphabet : str
            Alphabet.
        end_state : bool
            End state.
        """
        super(SilentState, self).__init__(name, alphabet, end_state)

    def emit(self, random):
        del random
        return ""

    def prob(self, seq: str, log_space: bool = False):
        """
        Parameters
        ----------
        seq : str
            Sequence.
        log_space : bool
            ``True`` to return in log space. Defaults to ``False``.
        """
        if seq == "":
            v = LOG(1.0)
        else:
            v = LOG(0.0)
        if not log_space:
            v = exp(v)
        return v

    def emission(self, log_space=False):
        if log_space:
            return [("", LOG(1.0))]
        return [("", 1.0)]

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
        """
        Parameters
        ----------
        name : str
            Name.
        emission : dict
            Emission probabilities in log space.
        """
        alphabet = "".join(list(emission.keys()))
        normalize_emission(emission)
        self._emission = emission
        super(NormalState, self).__init__(name, alphabet, False)

    def emit(self, random):
        probs = [exp(self._emission[a]) for a in self._alphabet]
        return random.choice(list(self._alphabet), p=probs)

    def prob(self, seq: str, log_space: bool = False):
        """
        Parameters
        ----------
        seq : str
            Sequence.
        log_space : bool
            ``True`` to return in log space. Defaults to ``False``.
        """
        v = self._emission.get(seq, LOG(0.0))
        if not log_space:
            v = exp(v)
        return v

    def emission(self, log_space=False):
        return emission_table(self._emission, log_space)

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
        """
        Parameters
        ----------
        name : str
            Name.
        alphabet : str
            Alphabet.
        emission : dict
            Emission probabilities in log space.
        """
        normalize_emission(emission)
        self._emission = emission
        super(TripletState, self).__init__(name, alphabet, False)

    def emit(self, random):
        triplets = list(self._emission.keys())
        probs = [exp(v) for v in self._emission.values()]
        return random.choice(triplets, p=probs)

    def prob(self, seq: str, log_space: bool = False):
        """
        Parameters
        ----------
        seq : str
            Sequence.
        log_space : bool
            ``True`` to return in log space. Defaults to ``False``.
        """
        v = self._emission.get(seq, LOG(0.0))
        if not log_space:
            v = exp(v)
        return v

    def emission(self, log_space: bool = False):
        """
        Parameters
        ----------
        log_space : bool
            ``True`` to return in log space. Defaults to ``False``.
        """
        return emission_table(self._emission, log_space)

    @property
    def min_len(self):
        return 3

    @property
    def max_len(self):
        return 3

    def __repr__(self):
        return f"<{self.__class__.__name__}:{self._name}>"


def emission_table(emission: dict, log_space: bool):
    """
    Parameters
    ----------
    emission : dict
        Emission probabilities in log space.
    log_space : bool
        ``True`` to return the probabilities in log space.
    """
    table = list(emission.items())
    table = sorted(table, key=lambda x: -x[1])
    if not log_space:
        table = [(row[0], exp(row[1])) for row in table]
    return table
