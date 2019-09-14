from math import exp, isinf
from ._log import LOG
from ._state import State


class HMM:
    def __init__(self, alphabet: str):
        self._init_logps = {}
        self._states = {}
        self._trans = {}
        self._alphabet = alphabet

    def init_prob(self, name, log_space=False):
        v = self._init_logps.get(name, LOG(0.0))
        if not log_space:
            v = exp(v)
        return v

    @property
    def states(self):
        return self._states

    def trans(self, name_a: str, name_b: str, log_space: bool = False):
        """
        Parameters
        ----------
        name_a : str
            Source state name.
        name_b : str
            Destination state name.
        log_space : bool
            ``True`` to return in log space. Defaults to ``False``.
        """
        v = self._trans.get(name_a, {}).get(name_b, LOG(0.0))
        if not log_space:
            v = exp(v)
        return v

    def set_trans(self, name_a: str, name_b: str, logp: float):
        """
        Parameters
        ----------
        name_a : str
            Source state name.
        name_b : str
            Destination state name.
        logp : bool
            Transition probability in log space.
        """
        self._trans[name_a][name_b] = logp

    @property
    def alphabet(self):
        return self._alphabet

    def add_state(self, state: State, init_logp: float = LOG(0.0)):
        """
        Parameters
        ----------
        state
            Add state.
        init_logp : bool
            Probability, in log space, of being the initial state.
        """
        if state.name in self._states:
            raise ValueError(f"State {state.name} already exists.")

        if set(state.alphabet) != set(self.alphabet):
            raise ValueError(f"Alphabet mismatch.")

        self._states[state.name] = state
        self._trans[state.name] = {}
        self._init_logps[state.name] = init_logp

        return state

    def rename_state(self, old_name: str, new_name: str):
        if old_name not in self._states:
            raise ValueError(f"State name `{old_name}` does not exist.")

        if new_name in self._states:
            raise ValueError(f"State name `{new_name}` already exists.")

        self._states[new_name] = self._states.pop(old_name)
        self._states[new_name].name = new_name
        self._init_logps[new_name] = self._init_logps.pop(old_name)

        for k, v in self._trans.items():
            if old_name in v:
                v[new_name] = v.pop(old_name)
        self._trans[new_name] = self._trans.pop(old_name)

    def delete_state(self, name):
        if name not in self._states:
            raise ValueError(f"State name `{name}` does not exist.")

        del self._states[name]
        del self._init_logps[name]
        del self._trans[name]
        for v in self._trans.values():
            if name in v:
                del v[name]

    def normalize(self):
        self._normalize_trans()
        self._normalize_init_logps()

    def emit(self, random):
        curr_state = self._draw_initial_state(random)
        path = []
        while not curr_state.end_state:
            seq = curr_state.emit(random)
            path.append((curr_state, seq))
            curr_state = self._transition(curr_state, random)
        path += [(curr_state, curr_state.emit(random))]
        return [(p[0].name, p[1]) for p in path]

    def likelihood(self, seq: str, state_path: list, log_space: bool = False):
        if len(state_path) == 0:
            if len(seq) == 0:
                if log_space:
                    return LOG(1.0)
                return 1.0

            if log_space:
                return LOG(0.0)
            return 0.0

        self._assure_states_exist([i[0] for i in state_path])
        head = state_path[0]
        qt = self._states[head[0]]
        ft = head[1]
        if ft > len(seq):
            return 0.0
        logp = self.init_prob(qt.name, True) + qt.prob(seq[:ft], True)

        seq = seq[ft:]
        qt_1 = qt
        for head in state_path[1:]:
            qt = self._states[head[0]]
            ft = head[1]
            logp += qt.prob(seq[:ft], True) + self.trans(qt_1.name, qt.name, True)
            seq = seq[ft:]
            qt_1 = qt

        if log_space:
            return logp
        return exp(logp)

    def draw(self, filepath, emissions=0, init_prob=True, digits=3, view=False):
        from graphviz import Digraph

        graph = Digraph()

        for state in self._states.values():
            if state.end_state:
                shape = "doublecircle"
            else:
                shape = "circle"

            if init_prob:
                p = self.init_prob(state.name, log_space=False)
                p = round(p, digits)
                if p > 0:
                    state_label = f"{state.name}: {p}"
                else:
                    state_label = f"{state.name}"
            else:
                state_label = f"{state.name}"

            if emissions > 0:
                emission = state.emission(log_space=False)
                emission = emission[:emissions]
                label = _format_emission_table(emission, state_label, digits)
            else:
                label = state_label

            graph.node(state.name, label, shape=shape)

        for state0, trans in self._trans.items():
            for state1, logp in trans.items():
                p = exp(logp)
                p = round(p, digits)
                if p > 0:
                    graph.edge(state0, state1, label=f"{p}")

        graph.render(filepath, view=view)

    def viterbi(self, seq: str, log_space: bool = False):
        max_logp = LOG(0.0)
        best_path = []
        end_states = [q for q in self._states.values() if q.end_state]
        if len(end_states) == 0:
            raise ValueError("There is no ending state to perform Viterbi.")

        for qt in end_states:
            for ft in range(qt.min_len, qt.max_len + 1):
                tup = self._viterbi(seq, qt, ft)
                if tup[0] > max_logp:
                    max_logp = tup[0]
                    best_path = tup[1] + [(qt, ft)]

        best_path = [(qt.name, ft) for qt, ft in best_path]
        if log_space:
            return max_logp, best_path
        return exp(max_logp), best_path

    def _viterbi(self, seq: str, qt: State, ft: int):
        max_logp = LOG(0.0)
        best_path = []
        emission_prob = qt.prob(seq[len(seq) - ft :], True)
        if emission_prob == LOG(0.0):
            return max_logp, best_path

        for qt_1 in self._states.values():
            if qt_1.end_state:
                continue

            T = self.trans(qt_1.name, qt.name, True)
            if T == LOG(0.0):
                continue

            for ft_1 in range(qt_1.min_len, qt_1.max_len + 1):
                seq_end = len(seq) - ft
                tup = self._viterbi(seq[:seq_end], qt_1, ft_1)
                tup = (tup[0] + T + emission_prob, tup[1] + [(qt_1, ft_1)])

                if tup[0] > max_logp:
                    max_logp = tup[0]
                    best_path = tup[1]

        if len(seq) - ft == 0:
            v = emission_prob + self.init_prob(qt.name, True)
            if v > max_logp:
                max_logp = v
                best_path = []

        return max_logp, best_path

    def _draw_initial_state(self, random):
        names = self._init_logps.keys()
        probs = [exp(v) for v in self._init_logps.values()]
        name = random.choice(list(names), p=probs)
        return self._states[name]

    def _transition(self, state: State, random):
        trans = self._trans[state.name]
        names = trans.keys()
        probs = [exp(v) for v in trans.values()]
        name = random.choice(list(names), p=probs)
        return self._states[name]

    def _normalize_trans(self):
        from scipy.special import logsumexp

        self._normalize_trans_end_states()

        names = self._states.keys()
        nstates = len(names)
        for a in names:
            logprobs = list(self._trans[a].values())

            if len(logprobs) == 0:
                for b in names:
                    self._trans[a][b] = -LOG(nstates)
            else:
                logprob_norm = logsumexp(logprobs)
                if isinf(logprob_norm):
                    for b in names:
                        self._trans[a][b] = -LOG(nstates)
                else:
                    for b in self._trans[a].keys():
                        self._trans[a][b] -= logprob_norm

    def _normalize_trans_end_states(self):
        state_names = self._states.keys()
        for state in self._states.values():
            if state.end_state:
                end_state_name = state.name
                for state_name in state_names:
                    self._trans[end_state_name][state_name] = LOG(0.0)
                self._trans[end_state_name][end_state_name] = LOG(1.0)

    def _normalize_init_logps(self):
        from scipy.special import logsumexp

        logprobs = list(self._init_logps.values())
        names = self._states.keys()
        nstates = len(names)

        logp_norm = logsumexp(logprobs)
        if isinf(logp_norm):
            for a in names:
                self._init_logps[a] = -LOG(nstates)
        else:
            for a in self._init_logps.keys():
                self._init_logps[a] -= logp_norm

    def _assure_states_exist(self, states):
        for state in states:
            if state not in self._states:
                raise ValueError(f"State `{state}` does not exist.")


def _format_emission_table(emission, name, digits):
    rows = ""
    for row in emission:
        seq = row[0]
        p = round(row[1], digits)
        if p == 0.0:
            break
        rows += f"<TR><TD>{seq}</TD><TD>{p}</TD></TR>"

    tbl_fmt = "BORDER='0' CELLBORDER='1' "
    tbl_fmt += "CELLSPACING='0' CELLPADDING='4'"
    tbl_str = f"<<TABLE {tbl_fmt}>"
    tbl_str += f"<TR><TD COLSPAN='2'>{name}</TD></TR>"
    tbl_str += rows
    tbl_str += "</TABLE>>"

    return tbl_str
