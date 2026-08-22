"""Ciclo 4: corrida completa Au/Be con el motor híbrido del ciclo 3."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CYCLE_3 = ROOT.parent / "cycle_3"
sys.path.insert(0, str(CYCLE_3))

import run_cycle_3 as engine
from phbands.inplane.media import InPlaneMedium


engine.ROOT = ROOT
engine.SYSTEM_LABEL = "Au/Be"
engine.FILLING_FRACTION = 0.0908
engine.CYCLE_NUMBER = 4
engine.OUTPUT_TAG = "cycle_4"
engine.HOST = InPlaneMedium(1.85, 13.11, 13.11 / 1.44)
engine.INCLUSION = InPlaneMedium(19.5, 3.36, 3.36 / 2.71)
engine.RADIUS = float(np.sqrt(engine.FILLING_FRACTION / np.pi))
engine.OMEGA_SCALE = 2 * np.pi * engine.HOST.c_t


if __name__ == "__main__":
    engine.main()
