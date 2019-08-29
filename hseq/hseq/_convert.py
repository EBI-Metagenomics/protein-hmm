_CodonBias = {"Y": {"TAT": 0.9, "TAC": 0.1}, "M": {"ATG": 1.0}}


def convert(bases, bases_emission, hmm):
    from ._hseq import HMM


#     aa_alphabet = hmm.alphabet
#     codon_hmm = HMM(create_frame_alphabet(bases))

#     states = hmm.states
#     init_probs = hmm.init_probs

#     codon_hmm.create_state("B", init_prob=init_probs["B"], silent=True)
#     codon_hmm.create_state("E", end_state=True, silent=True)

#     for i in range(1, 6):
#         M = codon_hmm.create_state(f"M{i}")
#         emission = states[f"M{i}"].emission
#         for j, aa in enumerate(hmm.alphabet):
#             create_frame_emission(bases, bases_emission, _CodonBias[aa], 0.1)
#         # M.set_emission()
#         pass

#     return codon_hmm


class ConvertProteinHMM:
    def __init__(self, epsilon: float, bg_base_emission: dict):
        self._epsilon = epsilon
        self._bases = "".join(bg_base_emission.keys())
        self._bg_base_emission = bg_base_emission
        self._frame_alphabet = {}
        self._create_frame_alphabet(5)

    def frame_emission(self, f, codon_emission):
        if f == 1:
            return self._frame1_emission(codon_emission)
        elif f == 2:
            return self._frame2_emission(codon_emission)
        elif f == 3:
            return self._frame3_emission(codon_emission)
        elif f == 4:
            return self._frame4_emission(codon_emission)
        elif f == 5:
            return self._frame5_emission(codon_emission)
        raise ValueError("Unknown frame length.")

    def length_probability(self, f):
        e = self._epsilon
        if f == 1:
            return e ** 2 * (1 - e) ** 2
        elif f == 2:
            return 2 * e ** 3 * (1 - e) + 2 * e * (1 - e) ** 3
        elif f == 3:
            return e ** 4 + 4 * e ** 2 * (1 - e) ** 2 + (1 - e) ** 4
        elif f == 4:
            return 2 * e ** 3 * (1 - e) + 2 * e * (1 - e) ** 3
        elif f == 5:
            return e ** 2 * (1 - e) ** 2
        raise ValueError("Unknown length.")

    def _create_frame_alphabet(self, length):
        if length == 1:
            alphabet = list(self._bases)
            self._frame_alphabet[1] = alphabet
            return alphabet

        abc = self._create_frame_alphabet(length - 1)
        alphabet = []
        for b in self._bases:
            for s in abc:
                alphabet.append(b + s)

        self._frame_alphabet[length] = alphabet
        return alphabet

    def _frame1_emission(self, codon_emission):
        """ p(Z=z1, F=1 | Q=M) """
        frames = self._frame_alphabet[1]
        emission = {}
        for frame in frames:
            z1 = frame[0]
            prob = 0.0
            for x2 in self._bases:
                for x3 in self._bases:
                    prob += codon_emission.get(z1 + x2 + x3, 0.0)
            emission[frame] = prob * self._epsilon ** 2 * (1 - self._epsilon) ** 2
        return emission

    def _frame2_emission(self, codon_emission):
        """ p(Z=z1z2, F=2 | Q=M) """
        frames = self._frame_alphabet[2]
        emission = {}
        for frame in frames:
            z1, z2 = frame

            prob = 0.0
            for x2 in self._bases:
                for x3 in self._bases:
                    prob += (
                        codon_emission.get(z1 + x2 + x3, 0.0)
                        * self._bg_base_emission[z2]
                    )
                    prob += (
                        codon_emission.get(z1 + x2 + x3, 0.0)
                        * self._bg_base_emission[z2]
                    )
            prob *= self._epsilon ** 3 * (1 - self._epsilon)

            e = self._epsilon * (1 - self._epsilon) ** 3
            for x3 in self._bases:
                prob += e * codon_emission.get(z1 + z2 + x3, 0.0)

            for x2 in self._bases:
                prob += e * codon_emission.get(z1 + x2 + z2, 0.0)

            emission[frame] = prob
        return emission

    def _frame3_emission(self, codon_emission):
        """ p(Z=z1z2z3, F=3 | Q=M) """
        frames = self._frame_alphabet[3]
        emission = {}
        for frame in frames:
            z1, z2, z3 = frame
            prob = 0.0

            e = self._epsilon ** 4
            for x2 in self._bases:
                for x3 in self._bases:
                    prob += (
                        e
                        * codon_emission.get(z1 + x2 + x3, 0.0)
                        * self._bg_base_emission[z2]
                        * self._bg_base_emission[z3]
                    )

            e = self._epsilon ** 2 * (1 - self._epsilon) ** 2
            for x3 in self._bases:
                prob += (
                    e
                    * codon_emission.get(z1 + z2 + x3, 0.0)
                    * self._bg_base_emission[z3]
                )

            e = self._epsilon ** 2 * (1 - self._epsilon) ** 2
            for x2 in self._bases:
                prob += (
                    e
                    * codon_emission.get(z1 + x2 + z3, 0.0)
                    * self._bg_base_emission[z2]
                )

            e = self._epsilon ** 2 * (1 - self._epsilon) ** 2
            for x3 in self._bases:
                prob += (
                    e
                    * codon_emission.get(z1 + z3 + x3, 0.0)
                    * self._bg_base_emission[z2]
                )

            e = self._epsilon ** 2 * (1 - self._epsilon) ** 2
            for x2 in self._bases:
                prob += (
                    e
                    * codon_emission.get(z1 + x2 + z3, 0.0)
                    * self._bg_base_emission[z2]
                )

            e = (1 - self._epsilon) ** 4
            prob += e * codon_emission.get(z1 + z2 + z3, 0.0)

            emission[frame] = prob
        return emission

    def _frame4_emission(self, codon_emission):
        """ p(Z=z1z2z3z4, F=4 | Q=M) """
        frames = self._frame_alphabet[4]
        emission = {}
        for frame in frames:
            z1, z2, z3, z4 = frame
            prob = 0.0

            e = self._epsilon ** 3 * (1 - self._epsilon)
            for x3 in self._bases:
                prob += (
                    e
                    * codon_emission.get(z1 + z3 + x3, 0.0)
                    * self._bg_base_emission[z2]
                    * self._bg_base_emission[z4]
                )

            e = self._epsilon ** 3 * (1 - self._epsilon)
            for x2 in self._bases:
                prob += (
                    e
                    * codon_emission.get(z1 + x2 + z4, 0.0)
                    * self._bg_base_emission[z2]
                    * self._bg_base_emission[z3]
                )

            e = self._epsilon * (1 - self._epsilon) ** 3
            prob += (
                e * codon_emission.get(z1 + z2 + z4, 0.0) * self._bg_base_emission[z3]
            )
            prob += (
                e * codon_emission.get(z1 + z3 + z4, 0.0) * self._bg_base_emission[z2]
            )

            emission[frame] = prob
        return emission

    def _frame5_emission(self, codon_emission):
        """ p(Z=z1z2z3z4z5, F=5 | Q=M) """
        frames = self._frame_alphabet[5]
        emission = {}
        for frame in frames:
            z1, z2, z3, z4, z5 = frame
            prob = (
                codon_emission.get(z1 + z3 + z5, 0.0)
                * self._bg_base_emission[z2]
                * self._bg_base_emission[z4]
            )

            emission[frame] = self._epsilon ** 2 * (1 - self._epsilon) ** 2 * prob
        return emission


# def create_frame_emission(bases, bases_emission, codon_emission, epsilon):
#     create_frame1_emission(bases, codon_emission, epsilon)
#     pass

