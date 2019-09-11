GENCODE = {
    "standard": {
        "F": ["UUU", "UUC"],
        "L": ["UUA", "UUG", "CUU", "CUC", "CUA", "CUG"],
        "I": ["AUU", "AUC", "AUA"],
        "M": ["AUG"],
        "V": ["GUU", "GUC", "GUA", "GUG"],
        "S": ["UCU", "UCC", "UCA", "UCG", "AGU", "AGC"],
        "P": ["CCU", "CCC", "CCA", "CCG"],
        "T": ["ACU", "ACC", "ACA", "ACG"],
        "A": ["GCU", "GCC", "GCA", "GCG"],
        "Y": ["UAU", "UAC"],
        "*": ["UAA", "UAG", "UGA"],
        "H": ["CAU", "CAC"],
        "Q": ["CAA", "CAG"],
        "N": ["AAU", "AAC"],
        "K": ["AAA", "AAG"],
        "D": ["GAU", "GAC"],
        "E": ["GAA", "GAG"],
        "C": ["UGU", "UGC"],
        "W": ["UGG"],
        "R": ["CGU", "CGC", "CGA", "CGG", "AGA", "AGG"],
        "G": ["GGU", "GGC", "GGA", "GGG"],
    }
}


def gencode(aa_or_codon, name="standard"):
    aa_or_codon = aa_or_codon.upper()
    if len(aa_or_codon) == 1:
        if aa_or_codon not in GENCODE[name]:
            aa = aa_or_codon
            raise ValueError(f"Amino acid `{aa}` was not found in the genetic code.")
        return GENCODE[name][aa_or_codon]
    else:
        aa_or_codon = aa_or_codon.replace("T", "U")
        for aa, codons in GENCODE[name].items():
            if aa_or_codon in codons:
                return aa
    codon = aa_or_codon
    raise ValueError(f"Codon `{codon}` was not found in the genetic code.")
