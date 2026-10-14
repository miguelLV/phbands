"""Estructura de bandas antiplanas para cavidades con pre-deformacion azimutal.

Matriz de caucho + obstaculos tipo cavidad cuyo entorno [r1, r2] quedo
pre-deformado azimutalmente (el borde interno giro un angulo psi). Usa el motor
del ciclo 6 con las correcciones del ciclo 9 ya activadas (matriz secular
simetrizada + parche de gauge de la suma de red).

ALCANCE: el T_n de este dispersor sale de la derivacion validada de
HANDOFF_azimuthal_predeformed_Tn_v2.md (Ogden small-on-large con las
convenciones de Parnell 2012). Es una FORMULA CERRADA en D, no una expansion
perturbativa; lo unico truncado es la fuente de la ecuacion de onda, que se
mantiene a O(D) descartando su termino D^2.

Consecuencia util: T_n(D) - T_n(0) resulta O(D^2) -- la correccion de primer
orden se anula exactamente -- y el defecto de unitariedad que deja el
truncamiento es O(D^3). Por eso el rango utilizable de psi es amplio; el script
reporta el defecto antes de calcular.

Sobre la incompresibilidad del caucho: para polarizacion ANTIPLANA solo entran
la densidad y el modulo de corte (c_t); la velocidad longitudinal no aparece en
ninguna parte del problema. Asumir el caucho incompresible (c_l -> infinito) es
por tanto compatible con este calculo sin necesidad de pasar c_l.

Ejemplos
--------
Corrida rapida de prueba (pocos puntos k, ventana corta)::

    python examples/run_azimuthal_bands.py --name prueba \\
        --k-per-segment 8 --n-scan 120 --omega-max 0.8 --psi 0.2

Comparacion sin deformar (psi=0) contra deformada, misma geometria::

    python examples/run_azimuthal_bands.py --name sin-deformar --psi 0.0
    python examples/run_azimuthal_bands.py --name deformada    --psi 0.3

Cada corrida escribe su carpeta en runs/ con figura, CSV y manifiesto.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Bandas antiplanas de cavidades con pre-deformacion azimutal en una "
            "red cuadrada de caucho."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--name", default="azimutal", help="Nombre del sistema (figuras y manifiesto)")

    # --- medio anfitrion (caucho, valores de la tesis; incompresible: c_l no interviene) ---
    p.add_argument("--host-rho", type=float, default=0.920, help="Densidad del caucho [g/cm^3]")
    p.add_argument("--host-ct", type=float, default=0.030, help="Velocidad transversal del caucho [km/s]")

    # --- geometria ---
    geom = p.add_mutually_exclusive_group()
    geom.add_argument("--filling-fraction", type=float, default=0.126,
                      help="Fraccion de area pi*(r1/a)^2 de la cavidad")
    geom.add_argument("--radius-ratio", type=float, help="Radio de la cavidad r1/a")
    p.add_argument("--r2-over-a", type=float, default=0.40,
                   help="Radio externo del anillo pre-deformado, r2/a (debe cumplir r1 < r2 < 0.5)")
    p.add_argument("--psi", type=float, default=0.2,
                   help="Angulo de rotacion del borde interno (parametro de pre-deformacion)")

    # --- numerica ---
    p.add_argument("--omega-max", type=float, default=1.3, help="Frecuencia normalizada maxima")
    p.add_argument("--k-per-segment", type=int, default=40, help="Puntos por tramo de M-Gamma-X-M")
    p.add_argument("--n-scan", type=int, default=520, help="Muestras del escaneo global en frecuencia")
    p.add_argument("--m-max", type=int, default=2, help="Corte modal cilindrico")
    p.add_argument("--h-primary", type=int, default=8, help="Corte primario de suma de red")
    p.add_argument("--h-check", type=int, default=12, help="Corte de validacion")
    p.add_argument("--h-pole", type=int, default=16, help="Corte adicional cerca de polos")
    p.add_argument("--no-gauge-patch", action="store_true",
                   help="Desactiva el parche de gauge de la suma de red (no recomendado)")
    p.add_argument("--output", type=Path, help="Directorio de salida; por defecto runs/<fecha>_<nombre>")
    return p


def main() -> None:
    args = parser().parse_args()

    if args.host_rho <= 0 or args.host_ct <= 0:
        raise ValueError("host-rho y host-ct deben ser positivos")
    if args.omega_max <= 0 or args.k_per_segment < 2 or args.n_scan < 20 or args.m_max < 0:
        raise ValueError("omega-max, k-per-segment, n-scan y m-max deben ser validos")

    if args.radius_ratio is not None:
        r1 = args.radius_ratio
        filling = float(np.pi * r1**2)
    else:
        if not 0 < args.filling_fraction < np.pi / 4:
            raise ValueError("filling-fraction debe estar entre 0 y pi/4")
        filling = args.filling_fraction
        r1 = float(np.sqrt(filling / np.pi))

    r2 = args.r2_over_a
    # El anillo pre-deformado es parte del dispersor: el formalismo MST supone
    # dispersores que no se solapan, asi que debe caber en el circulo inscrito
    # en la celda unidad (radio a/2).
    if not r1 < r2 < 0.5:
        raise ValueError(
            f"se requiere r1 < r2/a < 0.5 (los dispersores no pueden solaparse); "
            f"se recibio r1/a={r1:.5f}, r2/a={r2}"
        )

    output = (args.output
              or REPOSITORY / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_{args.name}").resolve()
    output.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(REPOSITORY / "cycles" / "cycle_6"))
    import run_cycle_6 as engine

    from phbands.media import HostMedium

    engine.ROOT = output
    engine.CYCLE_NUMBER = "azimutal"
    engine.OUTPUT_TAG = "azimuthal"
    engine.SYSTEM_LABEL = args.name
    engine.HOST_LABEL = "caucho"

    engine.SCATTERER_KIND = "cavity_azimuthal"
    engine.FILLING_FRACTION = filling
    engine.RADIUS = r1
    engine.AZIMUTHAL_R2 = r2
    engine.AZIMUTHAL_PSI = args.psi

    engine.HOST = HostMedium(mu=args.host_rho * args.host_ct**2, rho=args.host_rho)
    engine.OMEGA_SCALE = 2 * np.pi * engine.HOST.c_t
    engine.OMEGA_BAR_MAX = args.omega_max

    engine.N_PER_SEGMENT = args.k_per_segment
    engine.N_SCAN = args.n_scan
    engine.M_MAX = args.m_max
    engine.H_PRIMARY = args.h_primary
    engine.H_CHECK = args.h_check
    engine.H_POLE_CHECK = args.h_pole
    engine.HIGH_SYMMETRY_INDICES = [
        0, args.k_per_segment, 2 * args.k_per_segment, 3 * args.k_per_segment - 1
    ]

    # correcciones del ciclo 9
    engine.ADAPTIVE_MULTIPLICITY = True
    engine.SYMMETRIZED = True
    engine.GAUGE_ALTERNATIVES_OVER_A = () if args.no_gauge_patch else (0.45,)
    engine.BESSEL_POLE_GUARD = None

    D = args.psi * r1**2 / (r2**2 - r1**2)
    print(f"Salida     : {output}")
    print(f"Caucho     : rho={args.host_rho:.6g}, c_t={args.host_ct:.6g} (incompresible; c_l no interviene)")
    print(f"Cavidad    : r1/a={r1:.6g} (llenado {filling:.6g}), anillo hasta r2/a={r2:.6g}")
    print(f"Deformacion: psi={args.psi:.6g}  ->  D={D:.6g}")
    print(f"Numerica   : m_max={args.m_max}, n_scan={args.n_scan}, k/tramo={args.k_per_segment}")

    peor = unitarity_violation(engine, args)
    print(f"Validez    : defecto maximo de unitariedad |1+2T_n|-1 = {peor:.3%}")
    if peor > 0.05:
        print("  AVISO: con este psi el truncamiento de la fuente a O(D) ya pesa.")
        print("         Un dispersor sin perdidas cumple |1+2T_n| = 1 exactamente, y el")
        print("         caso sin deformar lo cumple; el defecto mide lo que se desprecio.")
        print("         Crece como D^3, asi que baja rapido: dividir psi por 2 lo divide")
        print(f"         por ~8. Para llegar a ~5% usa psi ~ {args.psi * (0.05 / peor) ** (1 / 3):.3g} o menor.")
    print()

    engine.main()


def unitarity_violation(engine, args) -> float:
    """Peor violacion de |1+2T_n| = 1 sobre la ventana de frecuencias y modos.

    Para un dispersor sin perdidas la matriz de dispersion 1+2T_n es unitaria.
    La pre-deformacion es estatica y no introduce perdidas, asi que la relacion
    debe seguir cumpliendose; el caso sin deformar la cumple de forma exacta. Lo
    que se mide aqui es, por tanto, cuanto se aparta de la fisica el haber
    truncado la fuente a O(D): es un termometro de validez, no un error
    numerico. Crece como D^3 (verificado en tests/test_cavity_azimuthal.py).
    """
    from phbands.scatterer import CavityAzimuthalPredeformedScatterer

    scatterer = CavityAzimuthalPredeformedScatterer(
        engine.RADIUS, engine.AZIMUTHAL_R2, engine.AZIMUTHAL_PSI, engine.HOST
    )
    peor = 0.0
    for fraction in (0.1, 0.25, 0.5, 0.75, 1.0):
        omega = fraction * args.omega_max * engine.OMEGA_SCALE
        for n in range(1, max(1, args.m_max) + 1):
            T = scatterer.T_n(omega, n)
            peor = max(peor, abs(abs(1.0 + 2.0 * T) - 1.0))
    return peor


if __name__ == "__main__":
    main()
