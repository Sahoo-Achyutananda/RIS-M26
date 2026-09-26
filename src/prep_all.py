"""Clean + feature-select several datasets, one subprocess per step so memory is released between them.

    python prep_all.py                         # all four
    python prep_all.py nsl_kdd unsw_nb15       # just these
Output: data/processed/<dataset>/dataset_final.csv
"""

import os
import subprocess
import sys

ALL = ["nsl_kdd", "unsw_nb15", "cic_ids2017", "cse_cic_ids2018"]
HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    datasets = sys.argv[1:] or ALL
    for ds in datasets:
        for script in ("preprocess.py", "feature_selection.py"):
            print(f"\n=== {ds}: {script} ===", flush=True)
            r = subprocess.run([sys.executable, "-u", script, "--dataset", ds], cwd=HERE)
            if r.returncode != 0:
                sys.exit(f"{ds}: {script} failed (exit {r.returncode}); later datasets not run")
