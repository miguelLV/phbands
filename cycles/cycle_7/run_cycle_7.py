"""Ciclo 7: Au/Be antiplano con multiplicidad adaptativa en M y Gamma."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CYCLE_6 = ROOT.parent / "cycle_6"
sys.path.insert(0, str(CYCLE_6))

import run_cycle_6 as engine


engine.ROOT = ROOT
engine.CYCLE_NUMBER = 7
engine.OUTPUT_TAG = "cycle_7"
engine.ADAPTIVE_MULTIPLICITY = True


if __name__ == "__main__":
    engine.main()
