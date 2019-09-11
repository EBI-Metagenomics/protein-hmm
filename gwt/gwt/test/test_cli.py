try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib

    pkg_resources = importlib.import_module("importlib_resources")

from click.testing import CliRunner

import gwt


def test_cli_frame():

    text = pkg_resources.read_text(gwt.test, "emission.txt")
    runner = CliRunner()
    with runner.isolated_filesystem():

        invoke = runner.invoke
        with open("emission.txt", "w") as f:
            f.write(text)

        cmd = ["frame", "emission.txt", "--epsilon", "0.1", "-n", "5"]
        r = invoke(gwt.cli, cmd)
        tbl = parse_table(r.stdout)
        assert abs(tbl["AAA"] - 0.04947252553677521) < 1e-6
        assert len(tbl) == 5

        cmd = [
            "frame",
            "emission.txt",
            "--show",
            "frame",
            "--epsilon",
            "0.1",
            "-n",
            "5",
        ]
        r = invoke(gwt.cli, cmd)
        tbl = parse_table(r.stdout)
        assert abs(tbl["AAA"] - 0.04947252553677521) < 1e-6
        assert abs(tbl["CAA"] - 0.025758667671559928) < 1e-6
        assert len(tbl) == 5

        cmd = [
            "frame",
            "emission.txt",
            "--show",
            "frame",
            "--epsilon",
            "1e-3",
            "-n",
            "10",
            "--base",
            "rna",
        ]
        r = invoke(gwt.cli, cmd)
        tbl = parse_table(r.stdout)
        assert abs(tbl["AAU"] - 0.029087254554619659) < 1e-6
        assert len(tbl) == 10


def test_cli_codon():

    text = pkg_resources.read_text(gwt.test, "emission.txt")
    runner = CliRunner()
    with runner.isolated_filesystem():

        invoke = runner.invoke
        with open("emission.txt", "w") as f:
            f.write(text)

        cmd = [
            "frame",
            "emission.txt",
            "--epsilon",
            "0.1",
            "-n",
            "3",
            "--show",
            "codon",
        ]
        r = invoke(gwt.cli, cmd)
        tbl = parse_table(r.stdout)
        assert abs(tbl["AAG"] - 0.07341651259277121) < 1e-6
        assert len(tbl) == 3

        cmd = [
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
        ]
        r = invoke(gwt.cli, cmd)
        tbl = parse_table(r.stdout)
        assert abs(tbl["GAT"] - 0.029729174721909286) < 1e-6
        assert len(tbl) == 10

        cmd = [
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
        ]
        r = invoke(gwt.cli, cmd)
        tbl = parse_table(r.stdout)
        assert abs(tbl["AAC"] - 0.02920380116025695) < 1e-6
        assert len(tbl) == 10


def test_cli_length():

    text = pkg_resources.read_text(gwt.test, "emission.txt")
    runner = CliRunner()
    with runner.isolated_filesystem():

        invoke = runner.invoke
        with open("emission.txt", "w") as f:
            f.write(text)

        cmd = ["frame", "emission.txt", "--epsilon", "1e-1", "--show", "length"]
        r = invoke(gwt.cli, cmd)
        # tbl = parse_table(r.stdout, " = ")
        # assert abs(tbl["p(F=1)"] - 0.0081) < 1e-6
        # assert abs(tbl["p(F=2)"] - 0.1476) < 1e-6
        # assert abs(tbl["p(F=3)"] - 0.6886) < 1e-6
        # assert abs(tbl["p(F=4)"] - 0.1476) < 1e-6
        # assert abs(tbl["p(F=5)"] - 0.0081) < 1e-6


# def test_cli_indels():

#     text = pkg_resources.read_text(gwt.test, "emission.txt")
#     runner = CliRunner()
#     with runner.isolated_filesystem():

#         invoke = runner.invoke
#         with open("emission.txt", "w") as f:
#             f.write(text)

#         cmd = ["frame", "emission.txt", "--epsilon", "1e-1", "--show", "indels"]
#         r = invoke(gwt.cli, cmd)
#         tbl = parse_table(r.stdout, " = ")
#         assert abs(tbl["p(M=0)"] - 0.6561) < 1e-6
#         assert abs(tbl["p(M=1)"] - 0.2916) < 1e-6
#         assert abs(tbl["p(M=2)"] - 0.0486) < 1e-6
#         assert abs(tbl["p(M=3)"] - 0.0036) < 1e-6
#         assert abs(tbl["p(M=4)"] - 0.0001) < 1e-6


def test_cli_gencode():
    invoke = CliRunner().invoke

    assert invoke(gwt.cli, ["gencode", "TGG"]).stdout.strip() == "W"
    # assert invoke(gwt.cli, ["gencode", "UGG"]).stdout.strip() == "W"
    # assert invoke(gwt.cli, ["gencode", "W"]).stdout.strip() == "UGG"
    # assert invoke(gwt.cli, ["gencode", "O"])


def parse_table(txt, sep=" "):
    txt = txt.strip()
    tbl = {}
    for line in txt.split("\n"):
        k, v = line.split(sep, 1)
        tbl[k.strip()] = float(v.strip())
    return tbl
