class State:
    def __init__(self, name: str, end_state: bool, alphabet: list, silent: bool):
        from numpy import ones

        self._name = name
        self._end_state = end_state
        self._emission = ones(len(alphabet)) / len(alphabet)
        self._alphabet = alphabet
        self._silent = silent

    @property
    def name(self):
        return self._name

    def set_emission(self, probs):
        from numpy import asarray

        probs = asarray(probs, float)
        if len(probs) != len(self._alphabet):
            err = "Number of emission probabilities is not equal to alphabet lenght."
            raise ValueError(err)

        self._emission[:] = probs

    @property
    def emission(self):
        return self._emission

    def emit(self, random):
        return random.choice(list(self._alphabet), p=self._emission)

    @property
    def silent(self):
        return self._silent

    @property
    def end_state(self):
        return self._end_state

    def __str__(self):
        return f"<{self._name}>"


class HMM:
    def __init__(self, alphabet: list):
        self._init_probs = {}
        self._states = {}
        self._trans = {}
        self._alphabet = alphabet

    @property
    def init_probs(self):
        return self._init_probs

    @property
    def states(self):
        return self._states

    @property
    def trans(self):
        return self._trans

    @property
    def alphabet(self):
        return self._alphabet

    def create_state(
        self,
        name: str,
        init_prob: float = 0.0,
        end_state: bool = False,
        silent: bool = False,
    ):
        if name in self._states:
            raise ValueError("State already exists.")

        state = State(name, end_state, alphabet=self._alphabet, silent=silent)
        self._states[state.name] = state
        self._trans[state.name] = {}
        self._init_probs[state.name] = init_prob

        return state

    def set_trans(self, a: str, b: str, prob: float):
        self._trans[a][b] = prob

    def normalize(self):
        self._set_all_trans()
        self._normalize_trans()
        self._normalize_init_probs()

    def emit(self, seed=0):
        from numpy.random import RandomState

        random = RandomState(seed)

        items = list(self._init_probs.items())
        state_names = [i[0] for i in items]
        probs = [i[1] for i in items]
        name = random.choice(state_names, p=probs)
        initial_state = self._states[name]

        visited_states = [initial_state]
        curr_state = initial_state
        sequence = []
        while not curr_state.end_state:
            if not curr_state.silent:
                sequence.append(curr_state.emit(random))

            curr_state = self._transition(curr_state, random)
            visited_states.append(curr_state)

        return "".join(str(s) for s in visited_states), "".join(sequence)

    def _transition(self, state: State, random):
        items = list(self._trans[state.name].items())
        state_names = [i[0] for i in items]
        probs = [i[1] for i in items]
        name = random.choice(state_names, p=probs)
        return self._states[name]

    def _set_all_trans(self):
        names = self._states.keys()
        for a in names:
            for b in names:
                if b not in self._trans[a]:
                    self.set_trans(a, b, 0.0)

    def _normalize_trans(self):
        names = self._states.keys()
        for a in names:
            prob_sum = sum(self._trans[a].values())
            if prob_sum == 0.0:
                for b in names:
                    self.set_trans(a, b, 1.0 / len(names))
            else:
                for b in names:
                    self.set_trans(a, b, self._trans[a][b] / prob_sum)

    def _normalize_init_probs(self):
        prob_sum = sum(self._init_probs.values())
        for a in self._init_probs.keys():
            self._init_probs[a] /= prob_sum


if __name__ == "__main__":
    import sys

    # Y Tyrosine
    #   UAU, UAC
    # M Methionine
    #   AUG

    seed = int(sys.argv[1])

    hmm = HMM("YM")

    hmm.create_state("B", init_prob=1.0, silent=True)
    hmm.create_state("E", end_state=True, silent=True)

    background_probs = [0.5, 0.5]
    match_probs = [0.2, 0.8]

    M1 = hmm.create_state("M1")
    M1.set_emission(match_probs)
    I1 = hmm.create_state("I1")
    I1.set_emission(background_probs)

    M2 = hmm.create_state("M2")
    M2.set_emission(match_probs)
    I2 = hmm.create_state("I2")
    I2.set_emission(background_probs)

    M3 = hmm.create_state("M3")
    M3.set_emission(match_probs)
    I3 = hmm.create_state("I3")
    I3.set_emission(background_probs)

    M4 = hmm.create_state("M4")
    M4.set_emission(match_probs)
    I4 = hmm.create_state("I4")
    I4.set_emission(background_probs)

    M5 = hmm.create_state("M5")
    M5.set_emission(match_probs)

    hmm.set_trans("B", "M1", 1.0)

    hmm.set_trans("M1", "M2", 0.8)
    hmm.set_trans("M1", "I1", 0.2)
    hmm.set_trans("I1", "I1", 0.9)
    hmm.set_trans("I1", "M2", 0.1)

    hmm.set_trans("M2", "M3", 0.8)
    hmm.set_trans("M2", "I2", 0.2)
    hmm.set_trans("I2", "I2", 0.9)
    hmm.set_trans("I2", "M3", 0.1)

    hmm.set_trans("M3", "M4", 0.8)
    hmm.set_trans("M3", "I3", 0.2)
    hmm.set_trans("I3", "I3", 0.9)
    hmm.set_trans("I3", "M4", 0.1)

    hmm.set_trans("M4", "M5", 0.8)
    hmm.set_trans("M4", "I4", 0.2)
    hmm.set_trans("I4", "I4", 0.9)
    hmm.set_trans("I4", "M5", 0.1)

    hmm.set_trans("M5", "E", 1.0)

    hmm.normalize()

    states, sequence = hmm.emit(seed)

    print(states)
    print(sequence)


