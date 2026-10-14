"""Ciclo 9: cavidad antiplana en caucho (nu=0.7) con las correcciones de
"Nivel 0" aplicadas sobre el estado del ciclo 7 (SIN la capa de certificacion
del ciclo 8).

Por que se vuelve al ciclo 7 en vez de seguir desde el 8
--------------------------------------------------------
Se midio la relacion costo/beneficio de la certificacion del ciclo 8 sobre este
mismo sistema (596 candidatos):

  * costo    : 737 s de 1052 s totales (70% del tiempo de ejecucion);
  * beneficio: rechaza 6 candidatos (1.0%) y deja 521 (87%) como "inciertos";
  * los 6 rechazos salen TODOS por `unstable_at_reciprocal_pole`, criterio que
    el motor del ciclo 6 ya aplica por su cuenta, gratis, en `search_windows`;
  * de los 147 candidatos que caen sobre un cero de Bessel (artefactos
    identificables analiticamente en microsegundos), la certificacion no
    rechaza NI UNO: 32 quedan "inciertos" y 1 queda "certificado";
  * las bandas certificadas no forman ramas completas sino 4 fragmentos
    arbitrarios del camino: no esta separando fisica de artefacto, esta
    cortando bandas.

Es decir, la capa mas cara del pipeline no discrimina. Las dos causas reales
del sobre-rechazo (matriz mal escalada y umbral de convergencia calibrado para
un m_max que no alcanza en llenado alto) se atacan mas barato y mas directo.

Que cambia respecto del ciclo 7
-------------------------------
1. `SYMMETRIZED = True`  -> el problema secular se evalua como
   I - T^{1/2} S T^{1/2}. El determinante es identico (se verifico: mismas
   raices a 5 decimales en varios k), asi que el escaneo global no cambia; lo
   que mejora es el condicionamiento de todo lo que usa SVD.

2. `BESSEL_POLE_GUARD = 1e-2` -> se activa la guarda de ceros de Bessel que la
   rama in-plane (ciclo 3) ya tenia y la rama antiplana nunca tuvo. Es el
   cambio de mayor impacto esperado: 147 de 596 candidatos (24.7%) de la
   corrida del ciclo 8 caen sobre un cero de J_{|l|+1}, agrupados en
   omega_bar = 0.6098, 0.8173, 1.0156, 1.1165 y 1.2077 -- exactamente las
   lineas planas espurias visibles en las figuras.

Ver docs/diagnostico_soluciones_espurias.md para el diagnostico completo con
la evidencia numerica y las referencias.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CYCLE_6 = ROOT.parent / "cycle_6"
sys.path.insert(0, str(CYCLE_6))

import run_cycle_6 as engine  # noqa: E402

from phbands.media import HostMedium  # noqa: E402


engine.ROOT = ROOT
engine.CYCLE_NUMBER = 9
engine.OUTPUT_TAG = "cycle_9"
engine.SYSTEM_LABEL = "cavidad-en-caucho"
engine.HOST_LABEL = "caucho"

# --- sistema fisico: cavidad cilindrica en caucho, llenado alto ---
engine.SCATTERER_KIND = "cavity"
engine.FILLING_FRACTION = 0.7
engine.RADIUS = float(np.sqrt(engine.FILLING_FRACTION / np.pi))
engine.HOST = HostMedium(mu=0.920 * 0.03**2, rho=0.920)
engine.OMEGA_SCALE = 2 * np.pi * engine.HOST.c_t
engine.OMEGA_BAR_MAX = 1.3
engine.N_SCAN = 520

# --- estado del ciclo 7 ---
engine.ADAPTIVE_MULTIPLICITY = True

# --- Nivel 0 ---
engine.SYMMETRIZED = True

# Parche de gauge local: elimina las singularidades ficticias de la suma de red
# EN SU ORIGEN, eligiendo por orden el radio auxiliar mejor condicionado. Con
# esto la guarda de Bessel deja de hacer falta y se apaga a proposito: era un
# filtro a posteriori, y ademas su clausula de escape por soporte de rama es
# contraproducente justamente para esta clase de artefacto (una linea plana a
# frecuencia constante tiene soporte de sobra, asi que se colaba igual: en la
# primera corrida del ciclo 9 sobrevivieron 81 de 147).
engine.GAUGE_ALTERNATIVES_OVER_A = (0.45,)
engine.BESSEL_POLE_GUARD = None


if __name__ == "__main__":
    engine.main()
