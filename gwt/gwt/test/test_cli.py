try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib

    pkg_resources = importlib.import_module("importlib_resources")
    # Try backported to PY<37 `importlib_resources`.

from click.testing import CliRunner
import gwt


def test_cli():

    text = pkg_resources.read_text(gwt.test, "emission.txt")
    runner = CliRunner()
    with runner.isolated_filesystem():

        invoke = runner.invoke
        with open("emission.txt", "w") as f:
            f.write(text)

        r = invoke(gwt.cli, ["frame", "emission.txt", "--epsilon", "0.1", "-n", "5"])
        tbl = parse_emission_table(r.stdout)
        assert abs(tbl["AAA"] - 0.049577130286817692) < 1e-6
        assert len(tbl) == 5

        r = invoke(
            gwt.cli,
            ["frame", "emission.txt", "--epsilon", "0.1", "-n", "3", "--show", "codon"],
        )
        tbl = parse_emission_table(r.stdout)
        assert abs(tbl["AAG"] - 0.073416512592771210) < 1e-6
        assert len(tbl) == 3

        r = invoke(
            gwt.cli,
            [
                "frame",
                "emission.txt",
                "--epsilon",
                "0.1",
                "-n",
                "10",
                "--show",
                "codon",
                "--base",
                "dna",
            ],
        )
        tbl = parse_emission_table(r.stdout)
        assert abs(tbl["AGT"] - 0.037957946563083392) < 1e-6
        assert len(tbl) == 10

        r = invoke(
            gwt.cli,
            [
                "frame",
                "emission.txt",
                "--epsilon",
                "0.1",
                "-n",
                "10",
                "--show",
                "codon",
                "--base",
                "rna",
            ],
        )
        tbl = parse_emission_table(r.stdout)
        assert abs(tbl["AGU"] - 0.037957946563083392) < 1e-6
        assert len(tbl) == 10

        assert invoke(gwt.cli, ["gencode", "TGG"]).stdout.strip() == "W"
        assert invoke(gwt.cli, ["gencode", "UGG"]).stdout.strip() == "W"
        assert invoke(gwt.cli, ["gencode", "W"]).stdout.strip() == "UGG"


def parse_emission_table(txt):
    txt = txt.strip()
    tbl = {}
    for line in txt.split("\n"):
        k, v = line.split(" ", 1)
        tbl[k.strip()] = float(v.strip())
    return tbl
