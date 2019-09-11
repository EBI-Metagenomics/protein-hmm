import click


@click.group(name="gwt", context_settings=dict(help_option_names=["-h", "--help"]))
def cli():
    """
    Show frame emission table.
    """
    pass


@click.command()
@click.argument("input", type=click.File("r"))
@click.option("--base", type=click.Choice(["dna", "rna"]), default="dna")
@click.option(
    "--show",
    type=click.Choice(["codon", "frame", "length", "indels"]),
    default="frame",
    help="Show codon emission P(X=𝚡₁𝚡₂𝚡₃), frame emission P(Z=𝚣₁𝚣₂..𝚣ₙ, F=n),"
    "length probability P(F=f), or indels probability p(M=m).",
)
@click.option("--epsilon", type=float, default=1e-3)
@click.option("--n", "-n", type=int, default=10, help="Number of table lines to show.")
def frame(input, base, show, epsilon, n):
    """
    Show frame emission information.

    INPUT is the file path to the amino acid emission table.
    One can also pass `-` to read from the standard input.
    """
    from ._parse import parse_emission
    from ._convert import AA2Codon
    from ._molecule import DNA, RNA
    from ._frame import FrameEmission

    aa_emission = parse_emission(input)
    if base == "dna":
        molecule = DNA()
    else:
        molecule = RNA()

    aa2codon = AA2Codon(aa_emission, molecule)
    if show == "codon":
        show_tuples(sort_emission(aa2codon.codon_emission().items())[:n], 3)
    elif show == "frame":
        frame = FrameEmission(aa2codon.codon_emission(False), molecule, epsilon)
        show_tuples(sort_emission(frame.top_emission(n)), 5)
    elif show == "length":
        frame = FrameEmission(aa2codon.codon_emission(False), molecule, epsilon)
        for f in range(1, 6):
            print(f"p(F={f}) = {frame.len_prob(f):.18f}")
    elif show == "indels":
        frame = FrameEmission(aa2codon.codon_emission(False), molecule, epsilon)
        for m in range(0, 5):
            print(f"p(M={m}) = {frame.indel_prob(m):.18f}")


@click.command()
@click.argument("aa_or_codon", type=str)
@click.option("--gencode", type=click.Choice(["standard"]), default="standard")
def gencode(aa_or_codon, gencode):
    """
    Translate amino acid to codons and vice versa.
    """
    from gwt import gencode as _gencode

    aa_or_codon = aa_or_codon.upper()
    try:
        print(" ".join(_gencode(aa_or_codon.upper(), gencode)))
    except ValueError as err:
        raise click.UsageError(str(err))


def sort_emission(emission):
    return list(sorted(emission, key=lambda x: x[1], reverse=True))


def show_tuples(d, length):
    for a, b in d:
        a += " " * (length - len(a))
        print(f"{a} {b:.18f}")


cli.add_command(frame)
cli.add_command(gencode)
