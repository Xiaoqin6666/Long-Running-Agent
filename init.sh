#!/usr/bin/env sh
set -eu

# Static bootstrap for the Long-Running Agent repository. Benchmark INIT
# scripts are generated separately under state/benchmarks/<benchmark_id>/init.sh.

python -m unittest discover -s tests
python -m compileall agent eval tests
python -m agent.main "Smoke test initialized long-running agent" --max-steps 5
