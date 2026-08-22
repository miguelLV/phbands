from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CYCLE_1 = ROOT.parent / "cycle_1"
FREQUENCY_TOLERANCE = 0.025


def main() -> None:
    with (CYCLE_1 / "candidate_validation_cycle_1.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["accepted"] == "True"]
    by_k: dict[int, list[float]] = {}
    for row in rows:
        by_k.setdefault(int(row["k_index"]), []).append(float(row["omega_bar_refined"]))

    gaps = []
    for index in range(1, 119):
        left = by_k.get(index - 1, [])
        center = by_k.get(index, [])
        right = by_k.get(index + 1, [])
        for left_frequency in left:
            matches = [value for value in right if abs(value - left_frequency) <= FREQUENCY_TOLERANCE]
            for right_frequency in matches:
                lower = min(left_frequency, right_frequency) - FREQUENCY_TOLERANCE
                upper = max(left_frequency, right_frequency) + FREQUENCY_TOLERANCE
                if not any(lower <= value <= upper for value in center):
                    gaps.append(
                        {
                            "k_index": index,
                            "left_frequency": left_frequency,
                            "right_frequency": right_frequency,
                            "search_lower_bar": max(0.0, lower),
                            "search_upper_bar": min(1.0, upper),
                        }
                    )

    # Un mismo hueco puede ser señalado por ramas muy próximas. Agrupar por
    # indice y ventanas solapadas, sin convertir su centro en una solucion.
    merged = []
    for gap in sorted(gaps, key=lambda item: (item["k_index"], item["search_lower_bar"])):
        if (
            merged
            and merged[-1]["k_index"] == gap["k_index"]
            and gap["search_lower_bar"] <= merged[-1]["search_upper_bar"]
        ):
            merged[-1]["search_lower_bar"] = min(merged[-1]["search_lower_bar"], gap["search_lower_bar"])
            merged[-1]["search_upper_bar"] = max(merged[-1]["search_upper_bar"], gap["search_upper_bar"])
            merged[-1]["evidence_count"] += 1
        else:
            merged.append({**gap, "evidence_count": 1})

    output = ROOT / "detected_gap_windows.json"
    output.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(json.dumps({"window_count": len(merged), "k_indices": sorted({item["k_index"] for item in merged}), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
