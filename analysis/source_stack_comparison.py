"""Source-stack comparison — Central Exeter network.

How each heat-source technology stack performs on the FIXED Central
Exeter tree topology and length (reused unchanged from
analysis/exeter_case_study.py), across duty, linear density, network
shape, and climate scenario. See analysis/source_stack_comparison_common.py
for the full methodology docstring (technology stacks, revenue basis,
GHNF grant, discount rate/lifetime) — this file is just the Exeter driver.

For the equivalent Ealing Town Centre study, see
analysis/source_stack_comparison_ealing.py.

Run from the repository root:
    python -m analysis.source_stack_comparison

Outputs CSVs and PNGs to output/source_stack_comparison/.
"""
from __future__ import annotations

from pathlib import Path

from analysis.exeter_case_study import CENTRAL_BUILDINGS, CENTRAL_SEGMENTS
from analysis.source_stack_comparison_common import run_study

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "source_stack_comparison"

if __name__ == "__main__":
    run_study("Central Exeter", CENTRAL_BUILDINGS, CENTRAL_SEGMENTS, OUT)
