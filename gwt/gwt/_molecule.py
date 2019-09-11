class Molecule:
    def __init__(self, bases):
        self._bases = bases

    @property
    def bases(self):
        return self._bases


class RNA(Molecule):
    def __init__(self):
        super().__init__("ACGU")

    @property
    def name(self):
        return "RNA"


class DNA(Molecule):
    def __init__(self):
        super().__init__("ACGT")

    @property
    def name(self):
        return "DNA"


def convert_to(sequence: str, molecule: Molecule):
    if molecule.name == "DNA":
        return sequence.replace("U", "T")
    elif molecule.name == "RNA":
        return sequence.replace("T", "U")
