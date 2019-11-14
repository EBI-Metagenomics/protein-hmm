# Protein HMM

We want to annotate error-prone nucleotide sequences in terms of proteins.

## Documents

The paper describing the theory can be found at [theory/](theory/).
We host the discussions at [trans-hmmer basecamp](https://3.basecamp.com/3983891/projects/14390790). The remaining documents and files are kept at the [Google-Drive:/Finn Team/Research/Translated search](https://drive.google.com/drive/u/1/folders/1VOSIZ7be9bUkqAG5ER-uaHM94vTj8G6_?ths=true) folder.

## Implementation

- [IMM](https://github.com/EBI-Metagenomics/imm): Invisible Markov Model library. It is an implementation of Hidden Markov Model whose states are allowed to emmit  variable-length sequences.
- [NMM](https://github.com/EBI-Metagenomics/nmm): Nucleotides Markov Model library. It defines IMMs with states that emmit sequences of nucleotides.
- [NMM-py](https://github.com/EBI-Metagenomics/nmm-py): Python package that builds nucleotide-aware Protein profiles.
- [Fasta reader](https://github.com/EBI-Metagenomics/fasta-reader-py)
- [HMMER reader](https://github.com/EBI-Metagenomics/hmmer-reader-py)

