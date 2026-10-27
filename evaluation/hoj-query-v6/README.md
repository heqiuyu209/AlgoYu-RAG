# New concrete examples on existing synthetic sources

24 assistant-authored cases, each split has eight supported and four unsupported
questions. Each split includes two cross-document comparisons. Sources are exact
copies of the same v2/v3 synthetic notes; this is not an unseen-document, external
or human-authored benchmark. No prior cases/ranks or implementation files were
supplied to the question writer. The writer received a one-sentence prefix-two
design summary while finishing, so this batch is not claimed to be fully blinded.

Queries and evidence were fixed before first ranking. Video labels were resolved
against actual chunks/times. Two labels for the same permutation chunk were merged
into one label requiring both original evidence spans, before freezing; no evidence
was dropped. Original all/heldout recall@3 >= 90% and nonregression gates apply.
