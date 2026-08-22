"""Ciclo 5: Au/Be de baja energia con rescate adaptativo de continuidad."""

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
engine.CYCLE_NUMBER = 5
engine.OUTPUT_TAG = "cycle_5"
engine.HOST = InPlaneMedium(1.85, 13.11, 13.11 / 1.44)
engine.INCLUSION = InPlaneMedium(19.5, 3.36, 3.36 / 2.71)
engine.RADIUS = float(np.sqrt(engine.FILLING_FRACTION / np.pi))
engine.OMEGA_SCALE = 2 * np.pi * engine.HOST.c_t

# La misma cantidad de muestras se concentra en la ventana fisica relevante.
engine.OMEGA_BAR_MAX = 0.5
engine.N_SCAN = 200

# El predictor solo activa ventanas: todas las raices se confirman con M.
engine.CONTINUITY_RECOVERY = True
engine.CONTINUITY_MATCH_TOLERANCE = 0.010

# Cerca de polos reciprocos se exige convergencia h=8 -> 12 -> 16 y soporte
# en puntos vecinos, en vez de descartar automaticamente la solucion.
engine.POLE_CHECK_H = 16
engine.RECIPROCAL_POLE_MIN_SUPPORT = 3

# Las raices fisicas recuperadas deben permanecer estables al aumentar h.
# Este umbral elimina minimos desplazados por un polo sin tocar el barrido.
engine.MAX_SHIFT_BAR = 5e-4


if __name__ == "__main__":
    engine.main()
