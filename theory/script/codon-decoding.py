from dataclasses import dataclass
from random import randint, seed
from typing import List, Tuple


@dataclass
class IndelStep:
    act: str
    pos: int
    lik: str

    def __repr__(self):
        return f"{self.act}:{self.pos}"


IndelPath = Tuple[IndelStep, ...]


def indel_tree_visit(path: IndelPath, seq_len: int, level: int) -> List[IndelPath]:
    if level == 4:
        return [path]

    Step = IndelStep
    paths: List[IndelPath] = []

    if level in [0, 1]:
        step = Step("N", 0, "(1-e)")
        paths += indel_tree_visit(path + (step,), seq_len, level + 1)
        for i in range(seq_len):
            step = Step("D", i, f"(e/{seq_len})")
            paths += indel_tree_visit(path + (step,), seq_len - 1, level + 1)

    elif level in [2, 3]:
        step = Step("N", 0, "(1-e)")
        paths += indel_tree_visit(path + (step,), seq_len, level + 1)
        for i in range(seq_len + 1):
            step = Step("I", i, f"(e/{seq_len+1})")
            paths += indel_tree_visit(path + (step,), seq_len + 1, level + 1)

    return paths


def generate_all_indel_paths():
    return indel_tree_visit(tuple(), 3, 0)


def apply_indel_step(step: IndelStep, seq: Tuple[str, ...]):
    if step.act == "N":
        return seq
    elif step.act == "I":
        return seq[: step.pos] + ("I",) + seq[step.pos :]

    assert step.act == "D"
    i = step.pos
    j = step.pos + 1
    return seq[:i] + seq[j:]


def apply_indel_path(path: IndelPath, seq: Tuple[str, str, str]):
    final_seq: Tuple[str, ...] = seq
    for step in path:
        final_seq = apply_indel_step(step, final_seq)
    return final_seq


def indel_path_prob(path: IndelPath):
    # ndels = sum(step.act == "D" for step in path)
    # nins = sum(step.act == "I" for step in path)
    # nnot = sum(step.act == "N" for step in path)
    # prob = ["(1-e)"] * nnot
    # return "*".join(prob)
    return "*".join(step.lik for step in path)


paths = generate_all_indel_paths()
print(f"Total number of paths: {len(paths)}")
seed(53)
for i in range(10):
    j = randint(0, len(paths) - 1)
    path = paths[j]
    seq = apply_indel_path(path, ("x1", "x2", "x3"))
    prob = indel_path_prob(path)
    print(f"{path}: {seq} -> p({prob})")


# class Sym:
#     def __init__(self, label: str):
#         self.label = label

#     def __repr__(self):
#         return self.label


# def _visit(seq: List[Sym], level: int):
#     seqs = []
#     if level in [0, 1]:
#         seqs += _visit(seq, level + 1)
#         for i in range(len(seq)):
#             j = i + 1
#             seqs += _visit(seq[:i] + seq[j:], level + 1)
#     elif level in [2, 3]:
#         seqs += _visit(seq, level + 1)
#         for i in range(len(seq) + 1):
#             seqs += _visit(seq[:i] + [Sym("I")] + seq[i:], level + 1)
#     else:
#         seqs.append(seq)
#     return seqs


# def generate_sequences():
#     return _visit([Sym("x1"), Sym("x2"), Sym("x3")], 0)


# def infer_ndels(seq: List[Sym]):
#     return 3 - sum(sym.label in ["x1", "x2", "x3"] for sym in seq)


# seqs = generate_sequences()
# for seq in seqs:
#     ndels = infer_ndels(seq)
#     print(f"{seq}: {ndels}")
