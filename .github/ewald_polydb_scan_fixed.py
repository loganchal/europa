#!/usr/bin/env python3
from __future__ import annotations
from fractions import Fraction
import numpy as np
import ewald_polydb_scan as scan


def parse_primitive_facets(raw, d: int) -> np.ndarray:
    rows = []
    for row in raw:
        if not isinstance(row, list):
            continue
        vals = [Fraction(str(x)) for x in row]
        if len(vals) != d + 1:
            raise ValueError(f"facet row has {len(vals)} entries, expected {d+1}")
        c = vals[0]
        if c == 0:
            raise ValueError("facet through origin")
        normal = [x / c for x in vals[1:]]
        if any(x.denominator != 1 for x in normal):
            raise ValueError(f"nonintegral normalized facet: {row}")
        rows.append([int(x) for x in normal])
    U = np.asarray(rows, dtype=np.int16)
    if U.ndim != 2 or U.shape[1] != d:
        raise ValueError("malformed facet matrix")
    return U


scan.parse_facets = parse_primitive_facets
scan.main()
