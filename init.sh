#!/usr/bin/env sh
set -eu

python -m unittest discover -s tests
python -m compileall agent eval tests
python -m agent.main "Smoke test initialized long-running agent" --max-steps 5

