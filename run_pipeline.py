"""One command to reproduce the full pipeline end to end.

    python3 run_pipeline.py

Runs, in order: normalize -> split -> block (train+test) -> features
(train+test) -> train -> threshold -> predict. Each phase is idempotent and
also runnable standalone (see its module docstring) — this just chains them
with the right arguments. Expect a few hours end-to-end on a machine
comparable to the one this was built on (10 cores / 16GB RAM): the feature
extraction steps dominate runtime (~100M + ~82M pairs at ~40-90k pairs/s,
single-process by design — see src/features.py for why).
"""

import subprocess
import sys
import time


STEPS = [
    ("normalize", [sys.executable, "-m", "src.normalize"]),
    ("split", [sys.executable, "-m", "src.split"]),
    ("block:train", [sys.executable, "-m", "src.blocking", "generate", "--split", "train"]),
    ("block:test", [sys.executable, "-m", "src.blocking", "generate", "--split", "test"]),
    ("features:train", [sys.executable, "-m", "src.features", "--split", "train"]),
    ("features:test", [sys.executable, "-m", "src.features", "--split", "test"]),
    ("train", [sys.executable, "-m", "src.train"]),
    ("threshold", [sys.executable, "-m", "src.threshold"]),
    ("evaluate", [sys.executable, "-m", "src.evaluate"]),
    ("predict", [sys.executable, "-m", "src.predict"]),
]


def main():
    t0 = time.time()
    for name, cmd in STEPS:
        print(f"\n{'='*70}\n[pipeline] {name}\n{'='*70}")
        step_t0 = time.time()
        subprocess.run(cmd, check=True)
        print(f"[pipeline] {name} done in {time.time()-step_t0:.1f}s")

    print(f"\n[pipeline] ALL DONE in {time.time()-t0:.1f}s")
    print("[pipeline] outputs: output/matching_results.tsv, output/candidate_pairs.tsv")
    print("[pipeline] validate with: python3 utils/validate_submission.py "
          "--matching output/matching_results.tsv --candidate output/candidate_pairs.tsv "
          "--test-dir dataset/test")


if __name__ == "__main__":
    main()
