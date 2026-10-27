# First-use public classroom PDF evaluation

Two original MIT OpenCourseWare 6.006 (Fall 2011) lecture PDFs, with exact text from the deployed service's bounded PDF parser. This is the first project evaluation using these two documents. The questions are written by a fresh source-only assistant, not students or an expert panel; this does not establish previously unseen model-training data or private classroom performance.

Source attribution: MIT OpenCourseWare; Erik Demaine and Srini Devadas. Source and extraction metadata are in `provenance.json`; original PDFs are retained unchanged. See `LICENSE.md`. No MIT endorsement is implied. Use is noncommercial isolated retrieval evaluation, without model training or automatic import into the site's knowledge base.

Twenty questions cover text, code conditions, complexity and page citations; calibration and heldout each contain eight supported and two unsupported questions. The author has no implementation, prior rankings or old evaluation context. Cases, exact evidence, physical-page labels, original PDFs, extracted text and metadata are frozen before the first query. After that first measurement, the dataset is exposed regression material.

Required evidence uses exact canonical extracted text. PDF layout losses are recorded separately in `extraction-observations.json`; successful retrieval does not establish complete mathematical formulas, diagram recovery, executable code reconstruction, grounded generation or refusal quality. Page numbering is physical PDF order, including the final attribution page.

The first run compares raw dense, legacy native FP32 selection and the currently deployed paired ONNX index. The existing recall, per-evidence protection, aggregate nonregression and same-run latency rules are unchanged. The prior 300-case regression is a separate immutable checkpoint, not part of this new 20-query process.
