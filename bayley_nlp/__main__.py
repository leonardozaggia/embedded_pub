"""
Unified command-line entrypoint for the three-step analytical pipeline
described in the paper's Methods section.

  python -m bayley_nlp step1   # Expert-guided translation (cascade B4->B3
                                # pairing + centroid-based assignment) ->
                                # 91-item theoretical measurement model
  python -m bayley_nlp step2   # Unsupervised consensus K-Means on the
                                # Bayley-4 39-item set, validated against
                                # the Aylward et al. (2022) reference models
  python -m bayley_nlp step3   # Unsupervised consensus K-Means on the full
                                # 91-item Bayley-III set, compared against
                                # the Step 1 translated structure
  python -m bayley_nlp all     # step1, then step2 and step3 (both compare
                                # their bottom-up clusters against Step 1's
                                # output, so step1 must run first)

Extra arguments are forwarded to the step module (see each module's --help),
e.g. `python -m bayley_nlp step2 --from-cache` re-runs only the reference-model
comparison on the committed embeddings and cluster assignments.

Each step is run as its own subprocess (rather than imported and called
in-process) because the underlying pipeline scripts compute their
configuration from module-level constants evaluated at import time; running
them as separate processes avoids any risk of one step's global state
leaking into another's within a single Python session.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

STEP_MODULES = {
    "step1": "bayley_nlp.pipelines.step1_translation",
    "step2": "bayley_nlp.pipelines.step2_b4_validation",
    "step3": "bayley_nlp.pipelines.step3_b3_bottomup",
}


def run_step(step: str, extra: list[str] | None = None) -> int:
    module = STEP_MODULES[step]
    cmd = [sys.executable, "-m", module] + list(extra or [])
    print(f"\n{'#' * 70}\n# Running {step} ({' '.join(cmd[1:])})\n{'#' * 70}\n", flush=True)
    result = subprocess.run(cmd)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bayley_nlp",
        description="Run the Bayley-III/Bayley-4 NLP factorization pipeline.",
    )
    parser.add_argument(
        "step",
        choices=["step1", "step2", "step3", "all"],
        help="Which pipeline step to run ('all' runs step1, then step2 and step3).",
    )
    parser.epilog = ("Any further arguments are passed through to the step module, e.g. "
                     "`python -m bayley_nlp step2 --from-cache`.")
    args, extra = parser.parse_known_args(argv)

    steps = ["step1", "step2", "step3"] if args.step == "all" else [args.step]
    for step in steps:
        code = run_step(step, extra)
        if code != 0:
            print(f"\n{step} failed with exit code {code}; stopping.", file=sys.stderr)
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
