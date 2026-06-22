# gmstrat

Code for sampling redistricting plans and constructing district/word strata.

## Setup

```bash
conda env create -f environment.yml
conda activate gmstrat
python -m pytest -q
```

## Julia

Sampling requires Julia 1.11+ and the environment in `sampling/runCycleWalkEnv`.

## Sampling

```bash
./sampling/run.sh \
  --map-file data/graph/grid_graph_5_by_5.json \
  --output-file output.jsonl.gz \
  --cycle-walk-steps 1e5 \
  --pop-dev 0.1
```

## Analysis

The Python modules provide sample processing, hierarchical district clustering, elbow and merge diagnostics, beam-search word construction, two-stage greedy pruning, and stationary/flux calculations. `demo.ipynb` runs the pipeline on the included 5-by-5 grid.

## Acknowledgements

Sampling is adapted from [CycleWalk.jl](https://github.com/jonmjonm/CycleWalk.jl). The HCC implementation follows [HCC-Ultra](https://arxiv.org/abs/2409.01010).
