"""Validacion posterior de candidatos reales del problema secular.

El determinante es sensible al escalamiento y puede presentar minimos falsos
cerca de polos de las sumas de red. Estas funciones usan el menor valor
singular de una matriz equilibrada y permiten refinar el candidato sin usar
el determinante como funcion objetivo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class MatrixCandidateValidation:
    omega: float
    balanced_sigma_min: float
    raw_sigma_min: float
    raw_sigma_max: float
    success: bool


def _equilibration_scales(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcula escalas de filas y columnas para reutilizarlas localmente."""
    working = np.asarray(matrix, dtype=complex)
    row_scale = np.max(np.abs(working), axis=1)
    row_scale = np.where(np.isfinite(row_scale) & (row_scale > 0), row_scale, 1.0)
    row_balanced = working / row_scale[:, None]
    column_scale = np.max(np.abs(row_balanced), axis=0)
    column_scale = np.where(np.isfinite(column_scale) & (column_scale > 0), column_scale, 1.0)
    return row_scale, column_scale


def equilibrate_matrix(
    matrix: np.ndarray,
    scales: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Equilibra filas y columnas por su norma infinita.

    El escalamiento es solo diagnostico: no cambia el rango de la matriz,
    pero evita que una fila o columna enorme domine la SVD numerica.
    """
    balanced = np.asarray(matrix, dtype=complex).copy()
    row_scale, column_scale = scales or _equilibration_scales(balanced)
    balanced /= row_scale[:, None]
    balanced /= column_scale[None, :]
    return balanced


def balanced_sigma_min(matrix: np.ndarray) -> float:
    """Menor valor singular despues de equilibrar filas y columnas."""
    if not np.all(np.isfinite(matrix)):
        return float("inf")
    return float(np.linalg.svd(equilibrate_matrix(matrix), compute_uv=False)[-1])


def refine_matrix_candidate(
    matrix_func: Callable[[float], np.ndarray],
    omega: float,
    half_window: float,
    refine_tol: float = 1e-9,
) -> MatrixCandidateValidation:
    """Refina localmente un candidato minimizando sigma_min equilibrado."""
    lower = max(1e-8, omega - half_window)
    upper = omega + half_window
    # Las escalas se fijan en el candidato. Recalcularlas en cada evaluacion
    # normalizaria tambien la direccion que tiende a cero y borraria el minimo.
    scales = _equilibration_scales(matrix_func(omega))

    def objective(value: float) -> float:
        matrix = matrix_func(value)
        if not np.all(np.isfinite(matrix)):
            return float("inf")
        return float(np.linalg.svd(equilibrate_matrix(matrix, scales), compute_uv=False)[-1])

    result = minimize_scalar(
        objective,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": refine_tol},
    )
    refined = float(result.x)
    matrix = matrix_func(refined)
    singular = np.linalg.svd(matrix, compute_uv=False)
    return MatrixCandidateValidation(
        omega=refined,
        balanced_sigma_min=objective(refined),
        raw_sigma_min=float(singular[-1]),
        raw_sigma_max=float(singular[0]),
        success=bool(result.success),
    )


def refine_with_expansion(
    matrix_func: Callable[[float], np.ndarray],
    omega: float,
    initial_half_window: float,
    max_expansions: int = 4,
    expansion_factor: float = 2.0,
    boundary_tol: float = 1e-3,
    refine_tol: float = 1e-9,
) -> tuple[MatrixCandidateValidation, float]:
    """Como `refine_matrix_candidate`, pero detecta cuando la ventana de
    busqueda era demasiado angosta y la ensancha en vez de devolver un
    resultado pegado al borde.

    `minimize_scalar(..., method="bounded")` con una ventana fija puede
    converger EXACTAMENTE al borde permitido cuando el minimo real esta
    mas alla de ese borde -- no porque no exista una raiz cercana, sino
    porque el optimizador esta forzado a quedarse dentro de los limites.
    Ese resultado no es un minimo genuino de sigma_min: tipicamente viene
    con sigma/residuo grandes, y arruina en cascada cualquier comparacion
    entre niveles que dependa de el (traslape modal, residuo prolongado).
    Se detecto empiricamente en la certificacion de raices antiplanas de
    cavidad (ver HANDOFF.md): >90% de los candidatos marcados "inciertos"
    tenian el desplazamiento pegado exactamente al ancho de la ventana de
    busqueda entre niveles.

    Devuelve `(resultado, half_window_usado)` -- el segundo valor sirve
    para diagnostico (cuantas veces hubo que ensanchar).
    """
    half_window = initial_half_window
    result = refine_matrix_candidate(matrix_func, omega, half_window, refine_tol)
    for _ in range(max_expansions):
        distance_from_center = abs(result.omega - omega)
        at_boundary = abs(distance_from_center - half_window) <= boundary_tol * half_window
        if not at_boundary:
            break
        half_window *= expansion_factor
        result = refine_matrix_candidate(matrix_func, omega, half_window, refine_tol)
    return result, half_window
