if __name__ == "__main__":
    from hseq import HMM
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


