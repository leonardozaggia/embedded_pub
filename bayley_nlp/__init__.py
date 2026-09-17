"""bayley_nlp -- NLP-based item factorization of the Bayley-III cognitive scale.

Implements the three-step pipeline described in the paper's Methods section:
cascade B4->B3 semantic pairing + centroid assignment (step1), unsupervised
consensus K-Means validation on the Bayley-4 (step2), and bottom-up
consensus K-Means on the full Bayley-III item set (step3). See
`bayley_nlp.pipelines` and `python -m bayley_nlp --help`.
"""

import sys

# The pipeline scripts print Unicode characters (arrows, etc.) in progress
# output. On Windows, stdout defaults to the system code page (commonly
# cp1252), which cannot encode them and crashes the run mid-pipeline.
# Forcing UTF-8 here -- which runs whenever any bayley_nlp submodule is
# invoked via `python -m` -- makes the pipeline's console output portable
# across platforms without changing what it prints.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")
del _stream
