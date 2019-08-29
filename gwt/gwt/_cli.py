import click


@click.command()
@click.argument("filepath", type=click.Path(exists=True, dir_okay=False))
def cli(filepath):
    """
    Show frame emission table.
    """
    from gwt import FrameEmission, AA2Codon

    epsilon = 0.01
    with open(filepath, "r") as fp:
        aa_emission = parse_aa_emission(fp)

    convert = AA2Codon(aa_emission, gencode="standard", molecule="DNA")

    fe = FrameEmission(convert.codon_emission(prob_space=True), epsilon)
    print(fe)


def parse_aa_emission(lines):

    aa_emission = {}
    for line in lines:
        aa, logp = line.split()
        logp = float(logp)
        aa_emission.update({aa: logp})

    return aa_emission
