# Public-source retrieval benchmark v8

Three unmodified Markdown articles by the **cp-algorithms contributors**, pinned
at revision `e12fc698882d468a0aaa183682f970da98b8415d`:

- [Manacher](https://github.com/cp-algorithms/cp-algorithms/blob/e12fc698882d468a0aaa183682f970da98b8415d/src/string/manacher.md)
- [Square Root Decomposition](https://github.com/cp-algorithms/cp-algorithms/blob/e12fc698882d468a0aaa183682f970da98b8415d/src/data_structures/sqrt_decomposition.md)
- [Linear Congruence Equation](https://github.com/cp-algorithms/cp-algorithms/blob/e12fc698882d468a0aaa183682f970da98b8415d/src/algebra/linear_congruence_equation.md)

The source articles and this derivative question set use **CC BY-SA 4.0**. See
[LICENSE](LICENSE) and [provenance.json](provenance.json) for exact source URLs,
download time, sizes and SHA-256 identities. Original article bytes are retained;
the manifest supplies stable evaluation identities and display titles. This is
an independent local evaluation, without endorsement by the original authors.

Questions are authored by a fresh source-only investigator from production
chunks before seeing retrieval results or implementation. Parent validation
checks that each labelled evidence group exists completely in one real source
chunk. Sixteen questions are split equally into calibration and heldout, each
with six supported and two unsupported questions. Frozen inputs must precede the
first model retrieval. This is a new-document **first measurement** only until
its results are observed; later use is exposed regression evidence. Public
articles are not private classroom material, and no generated-answer accuracy,
ASR accuracy, provider-model performance or refusal claim follows from retrieval.

First measurement completed on 2026-10-24: 43 chunks, 16 questions and 18 evidence
groups. Dense and the deployed-version CE both retrieved 15/18 evidence groups in
the first three results, below the unchanged 90% threshold; heldout was 9/10.
Three supported questions lacked complete top-three evidence. Four unsupported
questions still returned candidates, which does not measure generated refusals.
Current CE p95 also exceeded the original latency budget in this local sample.

The raw first measurement is retained at
`data/hoj-public-baseline-20261024/report.json`; source-only authorship, the one
pre-retrieval missing-dollar correction, and input identities are recorded in
[freeze.json](freeze.json). Later experiments use this dataset as exposed
regression evidence. No production changes were adopted from this measurement.
