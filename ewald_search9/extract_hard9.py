from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

from full_scan9 import (
    ALL,
    CANON_IDXS,
    D,
    INDEX,
    UNIT_IDXS,
    VECTORS,
    find_basis,
    iter_records,
    ray_mask,
    sha256,
)


def mask_for(facets):
    mask = ALL
    for a in facets[D:]:
        mask &= ray_mask(a)
    return mask


def scan(path, facets_count, top_n):
    by_complexity = []
    by_ecount = []
    forced = {}
    records = 0

    def push(heap, key, payload):
        item = (tuple(-x for x in key), payload["ordinal_zero_based"], payload)
        if len(heap) < top_n:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    for ordinal, facets in iter_records(path, facets_count):
        records += 1
        mask = mask_for(facets)
        ecount = mask.bit_count()
        units = sum((mask >> idx) & 1 for idx in UNIT_IDXS)
        payload = {
            "file": Path(path).name,
            "facets_count": facets_count,
            "ordinal_zero_based": ordinal,
            "ordinal_one_based": ordinal + 1,
            "units": units,
            "E_count": ecount,
            "facets": [[1, *a] for a in facets],
            "mask_hex": hex(mask),
        }
        push(by_complexity, (units, ecount, ordinal), payload)
        push(by_ecount, (ecount, units, ordinal), payload)
        if units <= 3:
            forced[ordinal] = payload
        if ordinal % 250000 == 0:
            print("PROGRESS", facets_count, ordinal, "units", units, "E", ecount, flush=True)

    selected = {p[2]["ordinal_zero_based"]: p[2] for p in by_complexity + by_ecount}
    selected.update(forced)
    result_cases = []
    for ordinal in sorted(selected):
        payload = selected[ordinal]
        mask = int(payload.pop("mask_hex"), 16)
        basis, meta = find_basis(mask)
        payload["basis"] = basis
        payload["basis_meta"] = meta
        payload["E_half"] = [VECTORS[i] for i in CANON_IDXS if (mask >> i) & 1]
        payload["selection"] = {
            "complexity": any(x[2]["ordinal_zero_based"] == ordinal for x in by_complexity),
            "small_E": any(x[2]["ordinal_zero_based"] == ordinal for x in by_ecount),
            "units_le_3": ordinal in forced,
        }
        result_cases.append(payload)

    result_cases.sort(key=lambda c: (c["units"], c["E_count"], c["ordinal_zero_based"]))
    return {
        "dimension": D,
        "file": str(path),
        "file_sha256": sha256(path),
        "facets_count": facets_count,
        "records": records,
        "top_n": top_n,
        "selected_count": len(result_cases),
        "cases": result_cases,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--facets", type=int, required=True)
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = scan(args.path, args.facets, args.top)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print("FINAL", json.dumps({
        "facets": args.facets,
        "records": result["records"],
        "selected": result["selected_count"],
        "out": args.out,
    }), flush=True)


if __name__ == "__main__":
    main()
