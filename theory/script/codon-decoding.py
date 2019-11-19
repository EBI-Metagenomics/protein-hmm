from typing import NamedTuple, List
from enum import Enum, auto


class Act(Enum):
    DEL = auto()
    INS = auto()
    NOT = auto()


IndelStep = NamedTuple("IndelStep", [("act", Act), ("pos", int)])


class Sym:
    def __init__(self, label: str):
        self.label = label

    def __repr__(self):
        return self.label


def _visit(seq: List[Sym], level: int):
    seqs = []
    if level in [0, 1]:
        seqs += _visit(seq, level + 1)
        for i in range(len(seq)):
            j = i + 1
            seqs += _visit(seq[:i] + seq[j:], level + 1)
    elif level in [2, 3]:
        seqs += _visit(seq, level + 1)
        for i in range(len(seq) + 1):
            seqs += _visit(seq[:i] + [Sym("I")] + seq[i:], level + 1)
    else:
        seqs.append(seq)
    return seqs


def generate_sequences():
    return _visit([Sym("x1"), Sym("x2"), Sym("x3")], 0)


def infer_ndels(seq: List[Sym]):
    return 3 - sum(sym.label in ["x1", "x2", "x3"] for sym in seq)


seqs = generate_sequences()
for seq in seqs:
    ndels = infer_ndels(seq)
    print(f"{seq}: {ndels}")
