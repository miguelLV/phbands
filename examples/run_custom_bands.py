"""Interfaz de terminal para explorar bandas fononicas 2D personalizadas."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]


def validate_medium(label: str, rho: float, c_l: float, c_t: float) -> float:
    if rho <= 0 or c_l <= 0 or c_t <= 0:
        raise ValueError(f"rho, c_l y c_t del {label} deben ser positivos")
    if c_l <= c_t:
        raise ValueError(f"c_l debe ser mayor que c_t en el {label}")
    return rho * c_t**2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Calcula bandas in-plane o antiplanas en una red cuadrada de cilindros.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    result.add_argument("--polarization", choices=("inplane", "antiplane"), default="inplane")
    result.add_argument("--name", default="custom", help="Nombre del sistema mostrado en figuras y manifiesto")
    result.add_argument("--host-rho", type=float, default=1.85, help="Densidad del anfitrion")
    result.add_argument("--host-cl", type=float, default=13.11, help="Velocidad longitudinal del anfitrion")
    result.add_argument("--host-ct", type=float, default=13.11 / 1.44, help="Velocidad transversal del anfitrion")
    result.add_argument("--inclusion-rho", type=float, default=19.5, help="Densidad del cilindro")
    result.add_argument("--inclusion-cl", type=float, default=3.36, help="Velocidad longitudinal del cilindro")
    result.add_argument("--inclusion-ct", type=float, default=3.36 / 2.71, help="Velocidad transversal del cilindro")
    geometry = result.add_mutually_exclusive_group()
    geometry.add_argument("--filling-fraction", type=float, default=0.0908, help="Fraccion de area pi*(r/a)^2")
    geometry.add_argument("--radius-ratio", type=float, help="Radio normalizado r/a")
    result.add_argument("--omega-max", type=float, default=0.5, help="Frecuencia normalizada maxima")
    result.add_argument("--k-per-segment", type=int, default=40, help="Puntos por tramo de M-Gamma-X-M")
    result.add_argument("--n-scan", type=int, default=200, help="Muestras del escaneo global en frecuencia")
    result.add_argument("--m-max", type=int, default=2, help="Corte modal cilindrico")
    result.add_argument("--h-primary", type=int, default=8, help="Corte primario de suma de red")
    result.add_argument("--h-check", type=int, default=12, help="Corte de validacion")
    result.add_argument("--h-pole", type=int, default=16, help="Corte adicional cerca de polos")
    result.add_argument("--output", type=Path, help="Directorio de salida; por defecto runs/<fecha>_<nombre>")
    return result


def configure_geometry(args):
    if args.radius_ratio is not None:
        if not 0 < args.radius_ratio < 0.5:
            raise ValueError("radius-ratio debe estar entre 0 y 0.5 para cilindros no solapados")
        radius = args.radius_ratio
        filling = float(np.pi * radius**2)
    else:
        if not 0 < args.filling_fraction < np.pi / 4:
            raise ValueError("filling-fraction debe estar entre 0 y pi/4 para una red cuadrada")
        filling = args.filling_fraction
        radius = float(np.sqrt(filling / np.pi))
    return filling, radius


def configure_common(engine, args, output, filling, radius):
    engine.ROOT = output
    engine.SYSTEM_LABEL = args.name
    engine.FILLING_FRACTION = filling
    engine.RADIUS = radius
    engine.OMEGA_BAR_MAX = args.omega_max
    engine.N_PER_SEGMENT = args.k_per_segment
    engine.N_SCAN = args.n_scan
    engine.M_MAX = args.m_max
    engine.H_PRIMARY = args.h_primary
    engine.H_CHECK = args.h_check
    engine.HIGH_SYMMETRY_INDICES = [0, args.k_per_segment, 2 * args.k_per_segment, 3 * args.k_per_segment - 1]
    engine.CYCLE_NUMBER = "personalizado"
    engine.OUTPUT_TAG = "custom"


def main():
    args = parser().parse_args()
    if args.omega_max <= 0 or args.k_per_segment < 2 or args.n_scan < 20 or args.m_max < 0:
        raise ValueError("omega-max, k-per-segment, n-scan y m-max deben ser validos y positivos")
    filling, radius = configure_geometry(args)
    host_mu = validate_medium("anfitrion", args.host_rho, args.host_cl, args.host_ct)
    inclusion_mu = validate_medium(
        "cilindro", args.inclusion_rho, args.inclusion_cl, args.inclusion_ct
    )
    output = (args.output or REPOSITORY / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_{args.name}").resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.polarization == "inplane":
        sys.path.insert(0, str(REPOSITORY / "cycles" / "cycle_3"))
        import run_cycle_3 as engine
        from phbands.inplane.media import InPlaneMedium

        configure_common(engine, args, output, filling, radius)
        engine.HOST = InPlaneMedium(args.host_rho, args.host_cl, args.host_ct)
        engine.INCLUSION = InPlaneMedium(args.inclusion_rho, args.inclusion_cl, args.inclusion_ct)
        engine.OMEGA_SCALE = 2 * np.pi * engine.HOST.c_t
        engine.POLE_CHECK_H = args.h_pole
        engine.CONTINUITY_RECOVERY = True
        engine.MAX_SHIFT_BAR = 5e-4
    else:
        sys.path.insert(0, str(REPOSITORY / "cycles" / "cycle_6"))
        import run_cycle_6 as engine
        from phbands.media import HostMedium

        configure_common(engine, args, output, filling, radius)
        engine.HOST = HostMedium(host_mu, args.host_rho)
        engine.INCLUSION = HostMedium(inclusion_mu, args.inclusion_rho)
        engine.OMEGA_SCALE = 2 * np.pi * engine.HOST.c_t
        engine.H_POLE_CHECK = args.h_pole
        engine.ADAPTIVE_MULTIPLICITY = True

    print(f"Salida: {output}")
    print(f"Host: rho={args.host_rho:.8g}, cl={args.host_cl:.8g}, ct={args.host_ct:.8g}")
    print(
        f"Inclusion: rho={args.inclusion_rho:.8g}, "
        f"cl={args.inclusion_cl:.8g}, ct={args.inclusion_ct:.8g}"
    )
    print(f"Geometria: filling={filling:.8g}, r/a={radius:.8g}")
    engine.main()


if __name__ == "__main__":
    main()
