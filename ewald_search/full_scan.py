from __future__ import annotations
import argparse, itertools, json, math, random, time
from pathlib import Path
import numpy as np

D = 8
VECTORS = [tuple(v) for v in itertools.product((-1, 0, 1), repeat=D)]
N = len(VECTORS)
INDEX = {v: i for i, v in enumerate(VECTORS)}
X = np.asarray(VECTORS, dtype=np.int16)
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
UNIT_IDXS = [
    INDEX[tuple(1 if i == j else 0 for i in range(D))]
    for j in range(D)
]

_RAY_CACHE = {}
_PROJ_CACHE = {}


def ray_mask(r):
    r = canon(tuple(map(int, r)))
    cached = _RAY_CACHE.get(r)
    if cached is not None:
        return cached
    a = np.asarray(r, dtype=np.int16)
    ok = np.abs(X @ a) <= 1
    raw = np.packbits(ok, bitorder="little").tobytes()
    mask = int.from_bytes(raw, "little") & ALL
    _RAY_CACHE[r] = mask
    return mask


def decode(s, base):
    n = int(s)
    digits = []
    while n:
        n, r = divmod(n, base)
        digits.append(r)
    if len(digits) < 2:
        raise ValueError("short record")
    dim, shift = digits[:2]
    vals = [z - shift for z in digits[2:]]
    if dim != D or len(vals) % D:
        raise ValueError((dim, shift, len(vals)))
    rays = [tuple(vals[i:i + D]) for i in range(0, len(vals), D)]
    eye = [tuple(1 if i == j else 0 for i in range(D)) for j in range(D)]
    if rays[:D] != eye:
        raise ValueError(("nonstandard initial facet", rays[:D]))
    return rays


def det_bareiss(rows):
    a = [list(map(int, r)) for r in rows]
    n = len(a)
    if n == 0:
        return 1
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


def rank_mod2(vecs, k):
    pivots = []
    for v in vecs:
        mask = sum((int(v[j]) & 1) << j for j in range(k))
        for p in pivots:
            mask = min(mask, mask ^ p)
        if mask:
            pivots.append(mask)
            pivots.sort(reverse=True)
    return len(pivots)


def projection_data(missing):
    missing = tuple(missing)
    cached = _PROJ_CACHE.get(missing)
    if cached is not None:
        return cached
    k = len(missing)
    pvecs = [
        tuple(p) for p in itertools.product((-1, 0, 1), repeat=k)
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


def primitive_prefix(rows, k):
    r = len(rows)
    if r == 0:
        return True
    g = 0
    for cols in itertools.combinations(range(k), r):
        minor = det_bareiss([[row[j] for j in cols] for row in rows])
        g = math.gcd(g, abs(minor))
        if g == 1:
            return True
    return False


def find_basis_projected(S, k, node_limit=20000):
    if k == 0:
        return []
    if rank_mod2(S, k) < k:
        return None
    order = sorted(S, key=lambda v: (sum(z != 0 for z in v), v))

    for off in range(min(len(order), 32)):
        rows = []
        for v in order[off:] + order[:off]:
            rr = rows + [v]
            r = len(rr)
            independent = any(
                det_bareiss([[row[j] for j in cols] for row in rr]) != 0
                for cols in itertools.combinations(range(k), r)
            )
            if independent:
                rows = rr
                if len(rows) == k:
                    if abs(det_bareiss(rows)) == 1:
                        return rows
                    break

    nodes = 0

    def rec(start, rows):
        nonlocal nodes
        nodes += 1
        if nodes > node_limit:
            return None
        if len(rows) == k:
            return rows if abs(det_bareiss(rows)) == 1 else None
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
        return ans

    rng = random.Random(hash(tuple(order)) & ((1 << 64) - 1))
    pool = order[:min(len(order), 250)]
    if len(pool) >= k:
        for _ in range(5000):
            rows = rng.sample(pool, k)
            if abs(det_bareiss(rows)) == 1:
                return rows
    return None


def find_basis(mask):
    present = [j for j, idx in enumerate(UNIT_IDXS) if (mask >> idx) & 1]
    if len(present) == D:
        return [VECTORS[i] for i in UNIT_IDXS], {"method": "units", "units": 8}

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

    proj_rank = rank_mod2(S, k)
    if proj_rank < k:
        return None, {
            "method": "rank2-obstruction", "units": len(present),
            "missing": missing, "proj_n": len(S), "proj_rank2": proj_rank,
        }

    rows = find_basis_projected(
        S, k, node_limit=50000 if k <= 5 else 120000
    )
    if rows is None:
        return None, {
            "method": "unresolved", "units": len(present),
            "missing": missing, "proj_n": len(S), "proj_rank2": k,
        }

    full = [
        tuple(1 if i == j else 0 for i in range(D)) for j in present
    ] + [reps[p] for p in rows]
    assert abs(det_bareiss(full)) == 1
    return full, {
        "method": "projected", "units": len(present),
        "missing": missing, "proj_n": len(S),
    }


def scan_file(path, start_id):
    lines = Path(path).read_text().splitlines()
    base = int(lines[0])
    unit_counts = {}
    unresolved = []
    best = []
    for off, s in enumerate(lines[1:]):
        if not s.strip():
            continue
        ID = start_id + off
        rays = decode(s.strip(), base)
        mask = ALL
        for ray in rays[D:]:
            mask &= ray_mask(ray)
        basis, meta = find_basis(mask)
        ecount = mask.bit_count()
        units = meta["units"]
        unit_counts[units] = unit_counts.get(units, 0) + 1
        best.append((units, ecount, len(rays), ID))
        best = sorted(best)[:25]

        if basis is None:
            E_half = [VECTORS[i] for i in CANON_IDXS if (mask >> i) & 1]
            out = {
                "id": ID, "block": Path(path).name, "offset": off,
                "rays": rays, "E_half": E_half, "E_count": ecount,
                "meta": meta,
            }
            unresolved.append(out)
            print("UNRESOLVED " + json.dumps(out, separators=(",", ":")), flush=True)
        elif off % 1000 == 0:
            print(
                "PROGRESS", Path(path).name, off, ID, "E", ecount,
                "units", units, "basis", basis, flush=True,
            )

    return {
        "file": str(path), "records": len(lines) - 1,
        "unit_counts": unit_counts, "best": best,
        "unresolved": unresolved,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out", default="ewald-result.json")
    args = ap.parse_args()
    root = Path(args.root)

    files = []
    for block in range(101):
        path = root / f"block{block}"
        if path.exists() and block % args.shards == args.shard:
            files.append((block, path))

    started = time.time()
    summaries = []
    unresolved = []
    total = 0
    for block, path in files:
        summary = scan_file(path, block * 7498 + 1)
        summaries.append(summary)
        unresolved.extend(summary["unresolved"])
        total += summary["records"]
        print(
            "BLOCK_DONE", block, "records", summary["records"],
            "unresolved", len(summary["unresolved"]),
            "best", summary["best"][:5], "ray_cache", len(_RAY_CACHE),
            "elapsed", round(time.time() - started, 2), flush=True,
        )

    result = {
        "shard": args.shard, "shards": args.shards,
        "records": total, "blocks": [b for b, _ in files],
        "unresolved": unresolved, "summaries": summaries,
        "ray_cache": len(_RAY_CACHE), "elapsed": time.time() - started,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print("FINAL " + json.dumps({
        "shard": args.shard, "records": total,
        "unresolved": len(unresolved), "ray_cache": len(_RAY_CACHE),
        "elapsed": round(time.time() - started, 2), "out": args.out,
    }), flush=True)


if __name__ == "__main__":
    main()
