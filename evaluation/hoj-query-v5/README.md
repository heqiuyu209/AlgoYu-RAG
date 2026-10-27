# Independent questions on existing synthetic sources

24 new questions: each calibration/heldout split contains 8 supported and 4
unsupported cases; four questions need evidence from multiple documents.
Sources are copied byte-for-byte from v2/v3, including text, Markdown, canonical
PDF pages and video transcripts. This is not an unseen-document or human benchmark.

A source-only agent authored the questions without prior cases, ranks or retrieval
code. Before any ranking, the root validated exact evidence against production
chunks, canonicalized CRT document IDs and video headings, and froze all inputs.
The original 90% all/heldout evidence recall@3 gate remains unchanged. Unsupported
cases diagnose scoring/answerability and do not assert generated-answer quality.
