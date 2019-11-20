from itertools import product
from dataclasses import dataclass
from typing import Any, List, Tuple

from sympy import Add, Mul, Symbol, simplify, Function
from sympy.functions.special.tensor_functions import KroneckerDelta


class I(KroneckerDelta):
    pass


@dataclass
class IndelStep:
    act: str
    pos: int
    lik: Any

    def __repr__(self):
        pos = str(self.pos)
        if self.act == "N":
            pos = "_"
        return f"{self.act}:{pos}"


IndelPath = Tuple[IndelStep, ...]

e = Symbol("e")
e1 = Symbol("(1-e)")

z1 = Symbol("z₁")
z2 = Symbol("z₂")
z3 = Symbol("z₃")
z4 = Symbol("z₄")
z5 = Symbol("z₅")

x1 = Symbol("x₁")
x2 = Symbol("x₂")
x3 = Symbol("x₃")

p = Function("p")

CODON = (x1, x2, x3)

ACTG = [0, 1, 2, 3]


def p_eval(x):
    if ACTG[0] == x:
        return 0.2
    if ACTG[1] == x:
        return 0.4
    if ACTG[2] == x:
        return 0.1
    assert ACTG[3] == x
    return 0.3


def indel_tree_visit(path: IndelPath, seq_len: int, level: int) -> List[IndelPath]:
    if level == 4:
        return [path]

    Step = IndelStep
    paths: List[IndelPath] = []

    if level in [0, 1]:
        step = Step("N", 0, e1)
        paths += indel_tree_visit(path + (step,), seq_len, level + 1)
        for i in range(seq_len):
            step = Step("D", i, e / seq_len)
            paths += indel_tree_visit(path + (step,), seq_len - 1, level + 1)

    elif level in [2, 3]:
        step = Step("N", 0, e1)
        paths += indel_tree_visit(path + (step,), seq_len, level + 1)
        for i in range(seq_len + 1):
            step = Step("I", i, e / (seq_len + 1))
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
    return Mul(*[step.lik for step in path])


def path_seq_len(path):
    return len(apply_indel_path(path, CODON))


def show_probs(paths):
    expr = []
    vecz = [z1, z2, z3, z4, z5]
    vecx = [x1, x2, x3]
    for path in paths:
        seq = apply_indel_path(path, CODON)
        path_prob = indel_path_prob(path)

        cond = []
        for i in range(len(seq)):
            z = vecz[i]
            if seq[i] == "I":
                cond.append(p(z))
                # cond.append(Symbol(f"p({str(z)})"))
            else:
                x = seq[i]
                cond.append(I(z, x))
                # cond.append(Symbol(f"I({z}={x})"))

        cond_prob = Mul(*cond)
        expr.append(cond_prob * path_prob)
        p1 = f"𝜋={path}"
        p2 = "𝐳={}".format("".join(str(i) for i in seq))
        zstr = "".join(str(z) for z in vecz[: len(seq)])
        xstr = "".join(str(x) for x in vecx)
        p3 = "p(𝛑=𝜋)={}".format(simplify(path_prob))
        p4 = f"p(Z={zstr}|X={xstr}, 𝛑=𝜋) = {simplify(cond_prob)}"
        p5 = f"p(Z={zstr}, 𝛑=𝜋|X={xstr}) = " + str(expr[-1])
        print(f"{p1} leads to {p2};")
        print(f"    {p3}; {p4};")
        print(f"    {p5}.")

    print()
    print(f"p(Z={zstr}|X=x₁x₂x₃) = " + str(simplify(Add(*expr))))

    return Add(*expr)


goal = """Let 𝜋 denote a path through the base-indel process.
We want to calculate p(Z=𝐳|X=x₁x₂x₃) = ∑_𝜋 p(Z=𝐳,𝛑=𝜋|X=x₁x₂x₃).

Note that p(Z=𝐳,𝛑=𝜋|X=x₁x₂x₃) = p(Z=𝐳|X=x₁x₂x₃,𝛑=𝜋)p(𝛑=𝜋|X=x₁x₂x₃)
                              = p(Z=𝐳|X=x₁x₂x₃,𝛑=𝜋)p(𝛑=𝜋)
"""
print(goal)

paths = generate_all_indel_paths()

# expr: List[Any] = []
expr = dict()
for i in range(1, 6):
    print(f"Paths that lead to |𝐳|={i}")
    print("------------------------")
    print()
    expr[f"|z|={i}"] = show_probs([path for path in paths if path_seq_len(path) == i])
    print()

# sexpr = simplify(Add(*expr))
# print(f"p(Z=𝐳|X=x₁x₂x₃) = " + str(sexpr))
print()

total = 0.0

zsyms = [z1, z2, z3, z4, z5]
for f in range(1, 6):
    for z in product(*[ACTG] * f):
        subs = [
            (e1, 1 - e),
            (e, 0.1),
            (x1, ACTG[1]),
            (x2, ACTG[0]),
            (x3, ACTG[3]),
        ]
        for i in range(f):
            subs += [(zsyms[i], z[i])]
        total += expr[f"|z|={f}"].subs(subs).replace(p, p_eval)
        # print(f"p(Z=z₁|X=x₁x₂x₃) = " + str(expr["|z|=1"].subs(subs).replace(p, p_eval)))
        # print(
        #     f"p(Z=z₁z₂|X=x₁x₂x₃) = " + str(expr["|z|=2"].subs(subs).replace(p, p_eval))
        # )
        # print(
        #     f"p(Z=z₁z₂z₃|X=x₁x₂x₃) = "
        #     + str(expr["|z|=3"].subs(subs).replace(p, p_eval))
        # )
        # print(
        #     f"p(Z=z₁z₂z₃z₄|X=x₁x₂x₃) = "
        #     + str(expr["|z|=4"].subs(subs).replace(p, p_eval))
        # )
        # print(
        #     f"p(Z=z₁z₂z₃z₄z₅|X=x₁x₂x₃) = "
        #     + str(expr["|z|=5"].subs(subs).replace(p, p_eval))
        # )
print(total)
