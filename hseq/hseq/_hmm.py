from math import inf, log, exp, isinf
from ._state import State


class HMM:
    def __init__(self, alphabet: str):
        self._init_probs = {}
        self._states = {}
        self._trans = {}
        self._alphabet = alphabet

    def init_prob(self, name, nlog_space=False):
        v = self._init_probs.get(name, inf)
        if not nlog_space:
            v = exp(-v)
        return v

    @property
    def states(self):
        return self._states

    def trans(self, name_a, name_b, nlog_space=False):
        v = self._trans.get(name_a, {}).get(name_b, inf)
        if not nlog_space:
            v = exp(-v)
        return v

    def set_trans(self, name_a: str, name_b: str, nlogp: float):
        self._trans[name_a][name_b] = nlogp

    @property
    def alphabet(self):
        return self._alphabet

    def add_state(self, state: State, init_prob: float = 0.0):
        if state.name in self._states:
            raise ValueError(f"State {state.name} already exists.")

        if set(state.alphabet) != set(self.alphabet):
            raise ValueError(f"Alphabet mismatch.")

        self._states[state.name] = state
        self._trans[state.name] = {}
        self._init_probs[state.name] = init_prob

        return state

    def normalize(self):
        self._normalize_trans()
        self._normalize_init_probs()

    # def emit(self, seed=0):
    #     from numpy.random import RandomState

    #     random = RandomState(seed)

    #     items = list(self._init_probs.items())
    #     state_names = [i[0] for i in items]
    #     probs = [i[1] for i in items]
    #     name = random.choice(state_names, p=probs)
    #     initial_state = self._states[name]

    #     visited_states = [initial_state]
    #     curr_state = initial_state
    #     sequence = []
    #     while not curr_state.end_state:
    #         if not curr_state.silent:
    #             sequence.append(curr_state.emit(random))

    #         curr_state = self._transition(curr_state, random)
    #         visited_states.append(curr_state)

    #     return "".join(str(s) for s in visited_states), "".join(sequence)

    # def _transition(self, state: State, random):
    #     items = list(self._trans[state.name].items())
    #     state_names = [i[0] for i in items]
    #     probs = [i[1] for i in items]
    #     name = random.choice(state_names, p=probs)
    #     return self._states[name]

    # def _set_all_trans(self):
    #     names = self._states.keys()
    #     for a in names:
    #         for b in names:
    #             if b not in self._trans[a]:
    #                 self.set_trans(a, b, 0.0)

    def _normalize_trans(self):
        from scipy.special import logsumexp

        names = self._states.keys()
        nstates = len(names)
        for a in names:
            probs = [-v for v in self._trans[a].values()]

            if len(probs) == 0:
                for b in names:
                    self._trans[a][b] = log(nstates)
            else:
                prob_sum = logsumexp(probs)
                if isinf(prob_sum):
                    for b in names:
                        self._trans[a][b] = log(nstates)
                else:
                    for b in self._trans[a].keys():
                        self._trans[a][b] += prob_sum

    def _normalize_init_probs(self):
        from scipy.special import logsumexp

        probs = [-v for v in self._init_probs.values()]
        names = self._states.keys()
        nstates = len(names)

        if len(probs) == 0:
            for a in names:
                self._init_probs[a] = log(nstates)
        else:
            prob_sum = logsumexp(probs)
            if isinf(prob_sum):
                for a in names:
                    self._init_probs[a] = log(nstates)
            else:
                for a in self._init_probs.keys():
                    self._init_probs[a] += prob_sum


