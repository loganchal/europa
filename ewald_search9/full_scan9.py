from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import random
import time
from pathlib import Path

import numpy as np

D = 9
VECTORS = [tuple(v) for v in itertools.product((-1, 0, 1), repeat=D)]
N = len(VECTORS)
INDEX = {v: i for i, v in enumerate(VECTORS)}
X = np.asarray(VECTORS, dtype=np.int64)
ALL = (1 << N) - 1


def canon(v):
    for z in v:
        if z:
            return tuple(v) if z > 0 else tuple(-x for x in v)
    return tuple(v)


CANON_IDXS = []
CANON_SELECT = 0
for i, v in enumerate(VECTORS):
    if any(v) and canon(v) == v:
        CANON_IDXS.append(i)
        CANON_SELECT |= 1 << i

UNIT_VECS = [tuple(1 if i == j else 0 for i in range(D)) for j in range(D)]
UNIT_IDXS = [INDEX[v] for v in UNIT_VECS]
STANDARD_FACETS = [tuple(-z for z in v) for v in UNIT_VECS]

_RAY_CACHE = {}
_PROJ_CACHE = {}


def ray_mask(ray):
    ray = canon(tuple(map(int, ray)))
    cached = _RAY_CACHE.get(ray)
    if cached is not None:
        return cached
    a = np.asarray(ray, dtype=np.int64)
    ok = np.abs(X @ a) <= 1
    raw = np.packbits(ok, bitorder="little").tobytes()
    mask = int.from_bytes(raw, "little") & ALL
    _RAY_CACHE[ray] = mask
    return mask


def det_bareiss(rows):
    a = [list(map(int, r)) for r in rows]
    n = len(a)
    if n == 0:
        return 1
    if any(len(r) != n for r in a):
        raise ValueError("determinant requires square matrix")
    sign = 1
    prev = 1
    for k in range(n - 1):
        if a[k][k] == 0:
            swap = next((i for i in range(k + 1, n) if a[i][k]), None)
            if swap is None:
                return 0
            a[k], a[swap] = a[swap], a[k]
            sign = -sign
        pivot = a[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                a[i][j] = (a[i][j] * pivot - a[i][k] * a[k][j]) // prev
            a[i][k] = 0
        prev = pivot
    return sign * a[-1][-1]


def rank_mod_p(vecs, k, p):
    rows = [[int(z) % p for z in v] for v in vecs]
    rank = 0
    for col in range(k):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][col]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inv = pow(rows[rank][col], -1, p)
        rows[rank] = [(x * inv) % p for x in rows[rank]]
        for i in range(len(rows)):
            if i != rank and rows[i][col]:
                c = rows[i][col]
                rows[i] = [(x - c * y) % p for x, y in zip(rows[i], rows[rank])]
        rank += 1
        if rank == k:
            return rank
    return rank


def projection_data(missing):
    missing = tuple(missing)
    cached = _PROJ_CACHE.get(missing)
    if cached is not None:
        return cached
    k = len(missing)
    pvecs = [
        tuple(p)
        for p in itertools.product((-1, 0, 1), repeat=k)
        if any(p) and canon(p) == tuple(p)
    ]
    pind = {p: i for i, p in enumerate(pvecs)}
    groups = [0] * len(pvecs)
    for idx in CANON_IDXS:
        v = VECTORS[idx]
        p = tuple(v[j] for j in missing)
        if not any(p):
            continue
        groups[pind[canon(p)]] |= 1 << idx
    cached = (pvecs, groups)
    _PROJ_CACHE[missing] = cached
    return cached


def maximal_minor_gcd(rows, k):
    r = len(rows)
    if r == 0:
        return 1
    g = 0
    for cols in itertools.combinations(range(k), r):
        minor = det_bareiss([[row[j] for j in cols] for row in rows])
        g = math.gcd(g, abs(minor))
        if g == 1:
            return 1
    return g


def primitive_prefix(rows, k):
    return maximal_minor_gcd(rows, k) == 1


def independent(rows, k):
    r = len(rows)
    return any(
        det_bareiss([[row[j] for j in cols] for row in rows]) != 0
        for cols in itertools.combinations(range(k), r)
    )


def find_basis_projected(S, k, node_limit):
    if k == 0:
        return [], {"search": "empty", "nodes": 0}

    order = sorted(set(S), key=lambda v: (sum(z != 0 for z in v), sum(abs(z) for z in v), v))

    for off in range(min(len(order), 64)):
        rows = []
        for v in order[off:] + order[:off]:
            rr = rows + [v]
            if independent(rr, k):
                rows = rr
                if len(rows) == k:
                    if abs(det_bareiss(rows)) == 1:
                        return rows, {"search": "greedy", "offset": off, "nodes": 0}
                    break

    rng = random.Random(hash(tuple(order)) & ((1 << 64) - 1))
    pool = order[: min(len(order), 500)]
    if len(pool) >= k:
        for trial in range(20000):
            rows = rng.sample(pool, k)
            if abs(det_bareiss(rows)) == 1:
                return rows, {"search": "random", "trial": trial + 1, "nodes": 0}

    nodes = 0
    aborted = False

    def rec(start, rows):
        nonlocal nodes, aborted
        nodes += 1
        if nodes > node_limit:
            aborted = True
            return None
        r = len(rows)
        if r == k:
            return rows if abs(det_bareiss(rows)) == 1 else None
        if len(order) - start < k - r:
            return None
        for pos in range(start, len(order)):
            rr = rows + [order[pos]]
            if not primitive_prefix(rr, k):
                continue
            ans = rec(pos + 1, rr)
            if ans is not None:
                return ans
        return None

    ans = rec(0, [])
    if ans is not None:
        return ans, {"search": "backtrack", "nodes": nodes}
    return None, {"search": "aborted" if aborted else "exhausted", "nodes": nodes}


def find_basis(mask):
    present = [j for j, idx in enumerate(UNIT_IDXS) if (mask >> idx) & 1]
    if len(present) == D:
        return UNIT_VECS, {"method": "units", "units": D}

    missing = tuple(j for j in range(D) if j not in present)
    k = len(missing)
    pvecs, groups = projection_data(missing)
    S = []
    reps = {}
    half = mask & CANON_SELECT
    for p, group in zip(pvecs, groups):
        intersection = half & group
        if not intersection:
            continue
        S.append(p)
        idx = (intersection & -intersection).bit_length() - 1
        v = VECTORS[idx]
        pv = tuple(v[j] for j in missing)
        reps[p] = v if pv == p else tuple(-z for z in v)

    ranks = {}
    for prime in (2, 3, 5, 7):
        rank = rank_mod_p(S, k, prime)
        ranks[str(prime)] = rank
        if rank < k:
            return None, {
                "method": "modular-rank-obstruction",
                "prime": prime,
                "units": len(present),
                "missing": missing,
                "proj_n": len(S),
                "ranks": ranks,
                "proved_failure": True,
            }

    if k <= 3:
        node_limit = 2_000_000
    elif k <= 5:
        node_limit = 500_000
    else:
        node_limit = 250_000
    rows, search_meta = find_basis_projected(S, k, node_limit=node_limit)
    if rows is None:
        return None, {
            "method": "unresolved",
            "units": len(present),
            "missing": missing,
            "proj_n": len(S),
            "ranks": ranks,
            "search": search_meta,
            "proved_failure": search_meta["search"] == "exhausted",
        }

    full = [UNIT_VECS[j] for j in present] + [reps[p] for p in rows]
    determinant = det_bareiss(full)
    assert abs(determinant) == 1
    for v in full:
        assert (mask >> INDEX[v]) & 1
    return full, {
        "method": "projected",
        "units": len(present),
        "missing": missing,
        "proj_n": len(S),
        "ranks": ranks,
        "search": search_meta,
        "determinant": determinant,
    }


def iter_records(path, expected_facets):
    current = None
    ordinal = -1
    with gzip.open(path, "rt", encoding="ascii", newline="") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if line == "FACETS":
                if current is not None:
                    raise ValueError(f"new record before blank at line {lineno}")
                current = []
                ordinal += 1
                continue
            if not line:
                if current is not None:
                    if len(current) != expected_facets:
                        raise ValueError(
                            f"record {ordinal}: {len(current)} facets, expected {expected_facets}"
                        )
                    yield ordinal, current
                    current = None
                continue
            if current is None:
                raise ValueError(f"data outside FACETS record at line {lineno}: {line[:80]}")
            row = tuple(map(int, line.split()))
            if len(row) != D + 1:
                raise ValueError(f"line {lineno}: row length {len(row)}")
            b, coeffs = row[0], row[1:]
            if b != 1:
                raise ValueError(f"line {lineno}: non-reflexive constant {b}")
            current.append(coeffs)
    if current is not None:
        if len(current) != expected_facets:
            raise ValueError(f"final record {ordinal}: wrong facet count {len(current)}")
        yield ordinal, current


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(path, expected_facets, out_path):
    started = time.time()
    unresolved = []
    methods = {}
    unit_counts = {}
    best = []
    records = 0

    for ordinal, facets in iter_records(path, expected_facets):
        records += 1
        if facets[:D] != STANDARD_FACETS:
            raise ValueError(
                f"record {ordinal}: first facets are not the standard coordinate facets"
            )

        mask = ALL
        for a in facets[D:]:
            mask &= ray_mask(a)

        basis, meta = find_basis(mask)
        ecount = mask.bit_count()
        units = meta["units"]
        method = meta["method"]
        methods[method] = methods.get(method, 0) + 1
        unit_counts[str(units)] = unit_counts.get(str(units), 0) + 1
        best.append((units, ecount, ordinal))
        best = sorted(best)[:50]

        if basis is None:
            E_half = [VECTORS[i] for i in CANON_IDXS if (mask >> i) & 1]
            case = {
                "id_guess": f"F.9D.f{expected_facets}.{ordinal:07d}",
                "file": Path(path).name,
                "ordinal_zero_based": ordinal,
                "ordinal_one_based": ordinal + 1,
                "facets": [[1, *a] for a in facets],
                "E_half": E_half,
                "E_count": ecount,
                "meta": meta,
            }
            unresolved.append(case)
            tag = "PROVED_FAILURE" if meta.get("proved_failure") else "UNRESOLVED"
            print(tag + " " + json.dumps(case, separators=(",", ":")), flush=True)
        elif ordinal % 100000 == 0:
            print(
                "PROGRESS", expected_facets, ordinal, "E", ecount,
                "units", units, "method", method,
                "basis", basis, "cache", len(_RAY_CACHE), flush=True,
            )

    result = {
        "dimension": D,
        "file": str(path),
        "file_sha256": sha256(path),
        "expected_facets": expected_facets,
        "records": records,
        "unresolved": unresolved,
        "methods": methods,
        "unit_counts": unit_counts,
        "best": best,
        "ray_cache": len(_RAY_CACHE),
        "elapsed_seconds": time.time() - started,
    }
    Path(out_path).write_text(json.dumps(result, indent=2))
    print("FINAL " + json.dumps({
        "facets": expected_facets,
        "records": records,
        "unresolved": len(unresolved),
        "proved_failures": sum(c["meta"].get("proved_failure", False) for c in unresolved),
        "methods": methods,
        "unit_counts": unit_counts,
        "best": best[:10],
        "ray_cache": len(_RAY_CACHE),
        "elapsed_seconds": round(result["elapsed_seconds"], 3),
        "out": str(out_path),
    }, separators=(",", ":")), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--facets", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    scan(args.path, args.facets, args.out)


if __name__ == "__main__":
    main()
