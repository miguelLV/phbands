"""Certificacion a posteriori de raices del problema secular antiplano.

Una raiz de una matriz truncada no se considera fisica solo porque su
determinante sea pequeno. Este modulo comprueba el residuo del vector nulo,
su estabilidad al ampliar la base multipolar y el residuo que deja el modo
truncado en las ecuaciones multipolares nuevas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class NullModeDiagnostics:
    sigma_min: float
    relative_residual: float
    edge_weight: float
    vector: np.ndarray


def null_mode_diagnostics(matrix: np.ndarray) -> NullModeDiagnostics:
    """Obtiene el vector menos singular y metricas invariantes de escala."""
    physical = np.asarray(matrix, dtype=complex)
    row_scale = np.max(np.abs(physical), axis=1)
    row_scale = np.where(np.isfinite(row_scale) & (row_scale > 0), row_scale, 1.0)
    row_balanced = physical / row_scale[:, None]
    column_scale = np.max(np.abs(row_balanced), axis=0)
    column_scale = np.where(np.isfinite(column_scale) & (column_scale > 0), column_scale, 1.0)
    balanced = row_balanced / column_scale[None, :]
    _, singular, vh = np.linalg.svd(balanced, full_matrices=False)
    # balanced = D_r M D_c y = 0; el vector fisico es x = D_c y.
    vector = vh[-1].conj() / column_scale
    vector /= max(np.linalg.norm(vector), np.finfo(float).tiny)
    denominator = max(np.linalg.norm(physical, 2), np.finfo(float).tiny)
    residual = float(np.linalg.norm(physical @ vector) / denominator)
    edge = float(abs(vector[0]) ** 2 + abs(vector[-1]) ** 2)
    return NullModeDiagnostics(float(singular[-1]), residual, edge, vector)


def central_projection(vector: np.ndarray, target_size: int) -> np.ndarray:
    """Proyecta una base [-M,M] sobre su bloque central de menor orden."""
    if target_size > len(vector) or (len(vector) - target_size) % 2:
        raise ValueError("Las bases multipolares deben ser impares y estar anidadas")
    offset = (len(vector) - target_size) // 2
    projected = np.asarray(vector[offset : offset + target_size], dtype=complex)
    norm = np.linalg.norm(projected)
    return projected / max(norm, np.finfo(float).tiny)


def modal_overlap(lower: np.ndarray, higher: np.ndarray) -> float:
    """Solapamiento de fase libre entre modos de dos cortes anidados."""
    lower_normalized = np.asarray(lower, dtype=complex)
    lower_normalized /= max(np.linalg.norm(lower_normalized), np.finfo(float).tiny)
    projected = central_projection(higher, len(lower_normalized))
    return float(np.clip(abs(np.vdot(lower_normalized, projected)), 0.0, 1.0))


def prolongated_residual(lower_vector: np.ndarray, higher_matrix: np.ndarray) -> float:
    """Residuo del modo bajo al exigir tambien las ecuaciones de orden alto."""
    if len(higher_matrix) < len(lower_vector):
        raise ValueError("La matriz de comprobacion debe tener una base mayor")
    padded = np.zeros(len(higher_matrix), dtype=complex)
    offset = (len(padded) - len(lower_vector)) // 2
    padded[offset : offset + len(lower_vector)] = lower_vector
    physical = np.asarray(higher_matrix, dtype=complex)
    denominator = max(np.linalg.norm(physical, 2), np.finfo(float).tiny)
    return float(np.linalg.norm(physical @ padded) / denominator)
