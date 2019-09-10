from math import log, exp, isinf
from ._nlog import nlog
from ._state import State


class HMM:
    def __init__(self, alphabet: str):
        self._init_probs = {}
        self._states = {}
        self._trans = {}
        self._alphabet = alphabet

    def init_prob(self, name, nlog_space=False):
        v = self._init_probs.get(name, nlog(0.0))
        if not nlog_space:
            v = exp(-v)
        return v

    @property
    def states(self):
        return self._states

    def trans(self, name_a, name_b, nlog_space=False):
        v = self._trans.get(name_a, {}).get(name_b, nlog(0.0))
        if not nlog_space:
            v = exp(-v)
        return v

    def set_trans(self, name_a: str, name_b: str, nlogp: float):
        self._trans[name_a][name_b] = nlogp

    @property
    def alphabet(self):
        return self._alphabet

    def add_state(self, state: State, init_prob: float = nlog(0.0)):
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

    def likelihood(self, seq: str, state_path: list):
        if len(state_path) == 0:
            if len(seq) == 0:
                return 1.0
            return 0.0
        head = state_path.pop(0)
        qt = self._states[head[0]]
        ft = head[1]
        if ft > len(seq):
            return 0.0
        p = self.init_prob(qt.name) * qt.prob(seq[:ft])

        seq = seq[ft:]
        qt_1 = qt
        for head in state_path:
            qt = self._states[head[0]]
            ft = head[1]
            p *= qt.prob(seq[:ft]) * self.trans(qt_1.name, qt.name)
            seq = seq[ft:]
            qt_1 = qt

        return p

    def draw(self, filepath, emission=False, init_prob=True, digits=3, view=False):
        from graphviz import Digraph

        graph = Digraph()

        for state in self._states.values():
            if state.end_state:
                shape = "doublecircle"
            else:
                shape = "circle"

            if init_prob:
                p = self.init_prob(state.name, nlog_space=False)
                if p > 0:
                    p = round(p, digits)
                    state_label = f"{state.name}: {p}"
                else:
                    state_label = f"{state.name}"
            else:
                state_label = f"{state.name}"

            if emission:
                emission = state.emission(nlog_space=False)
                label = _format_emission_table(emission, state_label, digits)
            else:
                label = state_label

            graph.node(state.name, label, shape=shape)

        for state0, trans in self._trans.items():
            for state1, nlogp in trans.items():
                p = exp(-nlogp)
                if p > 0:
                    p = round(p, digits)
                    graph.edge(state0, state1, label=f"{p}")

        graph.render(filepath, view=view)

    def viterbi(self, seq: str):
        max_prob = 0.0
        best_path = []
        end_states = [q for q in self._states.values() if q.end_state]
        if len(end_states) == 0:
            raise ValueError("There is no ending state to perform Viterbi.")
        for qt in end_states:
            for ft in range(qt.min_len, qt.max_len + 1):
                tup = self._viterbi(seq, qt, ft)
                if tup[0] > max_prob:
                    max_prob = tup[0]
                    best_path = tup[1] + [(qt, ft)]
        return max_prob, best_path

    def _viterbi(self, seq: str, qt: State, ft: int):
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
                tup = self._viterbi(seq[:seq_end], qt_1, ft_1)
                tup = (tup[0] * T * emission_prob, tup[1] + [(qt_1, ft_1)])

                if tup[0] > max_prob:
                    max_prob = tup[0]
                    best_path = tup[1]

        if len(seq) - ft == 0:
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
                    self._trans[end_state_name][state_name] = nlog(0.0)
                self._trans[end_state_name][end_state_name] = nlog(1.0)

    def _normalize_init_probs(self):
        from scipy.special import logsumexp

        probs = [-v for v in self._init_probs.values()]
        names = self._states.keys()
        nstates = len(names)

        prob_sum = logsumexp(probs)
        if isinf(prob_sum):
            for a in names:
                self._init_probs[a] = log(nstates)
        else:
            for a in self._init_probs.keys():
                self._init_probs[a] += prob_sum


def _format_emission_table(emission, name, digits):
    rows = ""
    for row in emission:
        if row[1] == 0.0:
            break
        seq = row[0]
        p = round(row[1], digits)
        rows += f"<TR><TD>{seq}</TD><TD>{p}</TD></TR>"

    tbl_fmt = "BORDER='0' CELLBORDER='1' "
    tbl_fmt += "CELLSPACING='0' CELLPADDING='4'"
    tbl_str = f"<<TABLE {tbl_fmt}>"
    tbl_str += f"<TR><TD COLSPAN='2'>{name}</TD></TR>"
    tbl_str += rows
    tbl_str += "</TABLE>>"

    return tbl_str
