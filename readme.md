# Requirements

## Python
`conda env create -f environment.yml && conda activate gmstrat`

## Julia

Requires Julia 1.11+ (recommended: `juliaup`):

- macOS/Linux: `curl -fsSL https://install.julialang.org | sh`
- Windows (PowerShell): `winget install --id Julialang.Juliaup -e`
- Then: `juliaup add 1.11 && juliaup default 1.11`

## Sampling
See the `Prepare` section in the demo notebook for how to generate grid json (containing graph information) needed for sampling.
`./sampling/run.sh --map-file data/graph/grid_graph_5_by_5.json --output-file local/output/grid5x5/atlas.jsonl.gz --cycle-walk-steps 1e5 --pop-dev 0.1`
The above samples 1e5 step, where a sample is dumped every 100 steps with a population deviation of 0.1 (i.e. ten percent). The frequency can be overriden with `cycle_walk_out_freq`. Check `run_cyclewalk.kl` for more options, including score selections.

## Analysis
Rest of the pipeline is contained in `demo.ipynb`.

## Acknowledgements
The sampling code is adapted from
`https://github.com/jonmjonm/CycleWalk.jl`
The hccfit implementation is adapted from
`https://arxiv.org/pdf/2409.01010` (I can't find the github repo.)