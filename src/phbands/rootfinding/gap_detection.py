"""Deteccion de posibles raices omitidas sin crear soluciones interpoladas.

Las frecuencias predichas por los extremos se usan exclusivamente para
activar una busqueda local posterior. Una prediccion nunca se devuelve como
raiz fisica ni se incorpora a la estructura de bandas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class GapTrigger:
    k_index: int
    frequency: float
    span: int
    left_index: int
    right_index: int
    reason: str = "one_to_one_bridge"


def detect_continuity_mismatches(
    bands: list[list[float]],
    max_span: int = 5,
    endpoint_delta_per_step: float = 0.035,
    match_tolerance: float = 0.010,
) -> list[GapTrigger]:
    """Activa ventanas cuando una rama esperada no tiene pareja cercana.

    A diferencia de :func:`detect_bridge_gaps`, no exige que disminuya el
    numero total de raices. Esto permite detectar una rama omitida cerca de
    un cruce aunque otra raiz conserve accidentalmente el conteo. Las
    frecuencias interpoladas son solo semillas para volver a evaluar M.
    """
    triggers: list[GapTrigger] = []
    for span in range(2, max_span + 1):
        for left_index in range(0, len(bands) - span):
            right_index = left_index + span
            pairs = _one_to_one_pairs(
                bands[left_index], bands[right_index], endpoint_delta_per_step * span
            )
            if not pairs:
                continue
            for k_index in range(left_index + 1, right_index):
                fraction = (k_index - left_index) / span
                expected = np.asarray(
                    [left + fraction * (right - left) for left, right in pairs], dtype=float
                )
                observed = np.asarray(bands[k_index], dtype=float)
                if observed.size:
                    cost = np.abs(expected[:, None] - observed[None, :])
                    rows, columns = linear_sum_assignment(cost)
                    matched = {
                        int(row)
                        for row, column in zip(rows, columns)
                        if cost[row, column] <= match_tolerance
                    }
                else:
                    matched = set()
                triggers.extend(
                    GapTrigger(
                        k_index,
                        float(value),
                        span,
                        left_index,
                        right_index,
                        reason="continuity_mismatch",
                    )
                    for position, value in enumerate(expected)
                    if position not in matched
                )
    return triggers


def _one_to_one_pairs(left: list[float], right: list[float], max_delta: float):
    if not left or not right:
        return []
    cost = np.abs(np.asarray(left)[:, None] - np.asarray(right)[None, :])
    rows, columns = linear_sum_assignment(cost)
    return [
        (float(left[row]), float(right[column]))
        for row, column in zip(rows, columns)
        if cost[row, column] <= max_delta
    ]


def detect_bridge_gaps(
    bands: list[list[float]],
    max_span: int = 5,
    endpoint_delta_per_step: float = 0.035,
    interior_tolerance: float = 0.012,
) -> list[GapTrigger]:
    """Detecta deficit de multiplicidad entre dos conjuntos de raices.

    Para cada par de extremos separados por 2..max_span indices, empareja sus
    raices uno-a-uno. En cada punto interior vuelve a resolver una asignacion
    uno-a-uno contra las frecuencias esperadas. Las expectativas no son
    soluciones: solo identifican ventanas que deben reexaminarse con M.
    """
    triggers: list[GapTrigger] = []
    for span in range(2, max_span + 1):
        for left_index in range(0, len(bands) - span):
            right_index = left_index + span
            pairs = _one_to_one_pairs(
                bands[left_index],
                bands[right_index],
                endpoint_delta_per_step * span,
            )
            if len(pairs) < 2:
                continue
            for k_index in range(left_index + 1, right_index):
                fraction = (k_index - left_index) / span
                expected = [left + fraction * (right - left) for left, right in pairs]
                observed = bands[k_index]
                # Con solo los extremos, la identidad de dos ramas que se
                # cruzan es ambigua. Activamos unicamente cuando falta
                # multiplicidad; si el conteo se conserva no inventamos una
                # asignacion de ramas basada en cercania.
                deficit = max(0, len(expected) - len(observed))
                if deficit == 0:
                    continue
                if observed:
                    cost = np.abs(np.asarray(expected)[:, None] - np.asarray(observed)[None, :])
                    rows, _ = linear_sum_assignment(cost)
                    matched_expected = {int(row) for row in rows}
                    unmatched = [
                        value for position, value in enumerate(expected) if position not in matched_expected
                    ]
                else:
                    unmatched = expected
                triggers.extend(
                    GapTrigger(k_index, float(value), span, left_index, right_index)
                    for value in unmatched
                )
    return triggers
