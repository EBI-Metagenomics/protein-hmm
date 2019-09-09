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

    def add_state(self, state: State, init_prob: float = inf):
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

    def emit(self, random):
        curr_state = self._draw_initial_state(random)
        path = []
        while not curr_state.end_state:
            seq = curr_state.emit(random)
            path.append((curr_state, seq))
            curr_state = self._transition(curr_state, random)
        return path + [(curr_state, curr_state.emit(random))]

    def viterbi(self, seq):
        max_prob = 0.0
        best_path = []
        for qt in self._states.values():
            for ft in range(qt.min_len, qt.max_len + 1):
                tup = self._viterbi(seq, qt, ft)
                if tup[0] > max_prob:
                    max_prob = tup[0]
                    best_path = tup[1] + [(qt, ft)]
        return max_prob, best_path

    def _viterbi(self, seq, qt, ft):
        max_prob = 0.0
        best_path = []
        emission_prob = qt.prob(seq[len(seq) - ft :])
        if emission_prob == 0.0:
            return max_prob, best_path

        for qt_1 in self._states.values():
            if qt_1.end_state:
                continue

            T = self.trans(qt_1.name, qt.name)
            if T == 0.0:
                continue

            for ft_1 in range(qt_1.min_len, qt_1.max_len + 1):
                seq_end = len(seq) - ft
                if seq_end < 0:
                    continue

                tup = self._viterbi(seq[:seq_end], qt_1, ft_1)
                tup = (tup[0] * T * emission_prob, tup[1] + [(qt_1, ft_1)])

                if tup[0] > max_prob:
                    max_prob = tup[0]
                    best_path = tup[1]

        if len(seq) - ft < 0:
            raise ValueError("Check this.")
        elif len(seq) - ft == 0:
            v = emission_prob * self.init_prob(qt.name)
            if v > max_prob:
                max_prob = v
                best_path = []

        return max_prob, best_path

    def _draw_initial_state(self, random):
        names = self._init_probs.keys()
        probs = [exp(-v) for v in self._init_probs.values()]

        name = random.choice(list(names), p=probs)
        return self._states[name]

    def _transition(self, state: State, random):
        trans = self._trans[state.name]
        names = trans.keys()
        probs = [exp(-v) for v in trans.values()]

        name = random.choice(list(names), p=probs)
        return self._states[name]

    def _normalize_trans(self):
        from scipy.special import logsumexp

        self._normalize_trans_end_states()

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

    def _normalize_trans_end_states(self):
        state_names = self._states.keys()
        for state in self._states.values():
            if state.end_state:
                end_state_name = state.name
                for state_name in state_names:
                    self._trans[end_state_name][state_name] = inf
                self._trans[end_state_name][end_state_name] = 0.0

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

