# Independent source-only question batch

24 new assistant-authored questions on the same ten synthetic notes: each split
contains eight supported and four unsupported queries, including two multi-source
queries. The fresh author received only source locations, schema and task rules,
without parent conversation, historical questions/ranks or algorithm descriptions.
This is not an unseen-document, external or human-authored benchmark.

Before first ranking, exact evidence and original page/time citations are checked
against production chunks and all inputs are frozen. PDF labels were normalized
from the author's "PDF物理第 N 页" to the existing parser's "第 N 页"; video labels
were resolved from original timestamps. No query or evidence span was changed.
