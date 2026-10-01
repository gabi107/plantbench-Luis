"""Running the Sobol' sample that the sensitivity analysis proposes.

    python -m studies.design.verify --workers 8

The analysis writes results/sobol-outcome.json, the configurations the Sobol' indices of
the realized outcome need, and this runs them into data/design-sobol.  The two steps are
separate because the runs are a dataset that outlives the analysis that asked for it and
is read back by run id.  The sample costs a closed-loop run per point where the damping
ratio costs a design solve.

The runs are written through `datagen.run_configurations`, so an interrupted run resumes
where it stopped and a run already recorded is not repeated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plantbench import datagen

from .analysis import SOBOL_DATA

HERE = Path(__file__).resolve().parent
# The runs the analysis reads: the full horizon of the search, and its features.
RUN = {"t_end": 1500.0, "dt": 2.0, "wall_budget": 600.0,
       "features": ["spectrum", "economics"], "states": False}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args(argv)
    proposal = args.results / "sobol-outcome.json"
    if not proposal.exists():
        raise SystemExit(f"{proposal} does not exist; run studies.design.analysis first")
    spec = json.loads(proposal.read_text())
    records = datagen.run_configurations(spec["case"], spec["configs"], SOBOL_DATA,
                                         run=RUN, workers=args.workers, label="sobol: ")
    print(f"\n{len(records)} runs written to {SOBOL_DATA}")


if __name__ == "__main__":
    main()
