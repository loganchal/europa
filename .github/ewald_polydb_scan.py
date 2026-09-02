#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import time
from collections import Counter

import numpy as np
from pymongo import MongoClient

URI = "mongodb://polymake:database@db.polymake.org:27017"
COLLECTION = "Polytopes.Lattice.SmoothReflexive"
D9_STRATA = {
    10: 1, 11: 91, 12: 3331, 13: 63971, 14: 583544,
    15: 2039665, 16: 2822309, 17: 1829247, 18: 666151,
    19: 173077, 20: 39218, 21: 7515, 22: 1324, 23: 226,
    24: 44, 25: 5, 26: 2,
}
ALLOWED = ["-1", "0", "1", -1, 0, 1]
OUTLIER = {
    "FACETS": {
        "$elemMatch": {
            "$elemMatch": {"$nin": ALLOWED}
        }
    }
}


def det_bareiss(A) -> int:
    a = [[int(x) for x in row] for row in A]
    n = len(a)
    sign = 1
    prev = 1
    for k in range(n - 1):
        if a[k][k] == 0:
            p = next((i for i in range(k + 1, n) if a[i][k]), None)
            if p is None:
                return 0
            a[k], a[p] = a[p], a[k]
            sign = -sign
        pivot = a[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                a[i][j] = (a[i][j] * pivot - a[i][k] * a[k][j]) // prev
        prev = pivot
        for i in range(k + 1, n):
            a[i][k] = 0
        for j in range(k + 1, n):
            a[k][j] = 0
    return sign * a[-1][-1]


def rank_mod(A: np.ndarray, p: int) -> int:
    M = (A.astype(np.int64) % p).copy()
    m, n = M.shape
    r = 0
    for j in range(n):
        rows = np.flatnonzero(M[r:, j])
        if not len(rows):
            continue
        i = r + int(rows[0])
        M[[r, i]] = M[[i, r]]
        inv = pow(int(M[r, j]), -1, p)
        M[r] = (M[r] * inv) % p
        for i in range(m):
            if i != r and M[i, j]:
                M[i] = (M[i] - M[i, j] * M[r]) % p
        r += 1
        if r == n:
            break
    return r


def canonical_vector(v):
    t = tuple(int(x) for x in v)
    for x in t:
        if x:
            return t if x > 0 else tuple(-y for y in t)
    return t


def canonical_normal(v):
    t = tuple(int(x) for x in v)
    nt = tuple(-x for x in t)
    return min(t, nt)


def parse_facets(raw, d: int) -> np.ndarray:
    rows = []
    for row in raw:
        if not isinstance(row, list):
            continue
        vals = [int(x) for x in row]
        if len(vals) != d + 1:
            raise ValueError(f"facet row has {len(vals)} entries, expected {d+1}")
        if vals[0] != 1:
            raise ValueError(f"facet constant is {vals[0]}, expected 1")
        rows.append(vals[1:])
    U = np.asarray(rows, dtype=np.int16)
    if U.ndim != 2 or U.shape[1] != d:
        raise ValueError("malformed facet matrix")
    return U


class EwaldEngine:
    def __init__(self, d: int, seed_catalogue: int = 768):
        self.d = d
        self.ternary = np.asarray(
            list(itertools.product((-1, 0, 1), repeat=d)), dtype=np.int16
        )
        self.npoints = len(self.ternary)
        self.full_mask = (1 << self.npoints) - 1
        self.index = {tuple(map(int, v)): i for i, v in enumerate(self.ternary)}
        self.ident = np.eye(d, dtype=np.int64)
        self.normal_masks: dict[tuple[int, ...], int] = {}
        self.emask_basis: dict[int, tuple[int, np.ndarray]] = {}
        self.catalogue: list[tuple[int, np.ndarray, int]] = []
        self.catalogue_seen: set[int] = set()
        self.stats = Counter()
        self._add_basis(self.ident)
        self._seed_catalogue(seed_catalogue)

    def _vector_bit(self, v) -> int:
        cv = canonical_vector(v)
        return 1 << self.index[cv]

    def _basis_mask(self, A: np.ndarray) -> int:
        mask = 0
        for row in A:
            mask |= self._vector_bit(row)
        return mask

    def _add_basis(self, A: np.ndarray):
        A = np.asarray(A, dtype=np.int64)
        if A.shape != (self.d, self.d):
            raise ValueError("bad basis shape")
        if abs(det_bareiss(A.tolist())) != 1:
            raise ValueError("catalogue matrix is not unimodular")
        if np.max(np.abs(A)) > 1:
            return
        bmask = self._basis_mask(A)
        if bmask in self.catalogue_seen:
            return
        self.catalogue_seen.add(bmask)
        self.catalogue.append((bmask, A.copy(), 0))

    def _seed_catalogue(self, target: int):
        rng = random.Random(0xE9A1D)
        attempts = 0
        while len(self.catalogue) < target and attempts < target * 200:
            attempts += 1
            A = np.eye(self.d, dtype=np.int64)
            for _ in range(rng.randint(self.d, 5 * self.d)):
                op = rng.randrange(4)
                if op == 0:
                    i, j = rng.sample(range(self.d), 2)
                    A[[i, j]] = A[[j, i]]
                elif op == 1:
                    i = rng.randrange(self.d)
                    A[i] *= -1
                else:
                    i, j = rng.sample(range(self.d), 2)
                    s = -1 if rng.randrange(2) else 1
                    candidate = A[i] + s * A[j]
                    if np.max(np.abs(candidate)) <= 1:
                        A[i] = candidate
            self._add_basis(A)

    def constraint_mask(self, normal) -> int:
        key = canonical_normal(normal)
        cached = self.normal_masks.get(key)
        if cached is not None:
            return cached
        c = np.asarray(key, dtype=np.int64)
        ok = np.abs(self.ternary.astype(np.int64) @ c) <= 1
        packed = np.packbits(ok.astype(np.uint8), bitorder="little").tobytes()
        mask = int.from_bytes(packed, "little") & self.full_mask
        self.normal_masks[key] = mask
        return mask

    def ewald_mask(self, U: np.ndarray) -> int:
        mask = self.full_mask
        for u in U:
            mask &= self.constraint_mask(u)
        return mask

    def mask_to_points(self, mask: int) -> np.ndarray:
        idx = []
        m = mask
        while m:
            bit = m & -m
            idx.append(bit.bit_length() - 1)
            m ^= bit
        return self.ternary[np.asarray(idx, dtype=np.int64)]

    def _catalogue_lookup(self, emask: int):
        hit = self.emask_basis.get(emask)
        if hit is not None:
            self.stats["emask_cache"] += 1
            return hit[1].copy(), "emask_cache"
        for pos, (bmask, A, uses) in enumerate(self.catalogue):
            if bmask & ~emask == 0:
                self.catalogue[pos] = (bmask, A, uses + 1)
                if pos > 8 and uses > 0:
                    item = self.catalogue.pop(pos)
                    self.catalogue.insert(1, item)
                self.stats["catalogue"] += 1
                self.emask_basis[emask] = (bmask, A.copy())
                return A.copy(), "catalogue"
        return None, None

    def _sign_reps(self, E: np.ndarray) -> np.ndarray:
        out = []
        for v in E:
            if np.all(v == 0):
                continue
            cv = canonical_vector(v)
            if tuple(map(int, v)) == cv:
                out.append(v)
        return np.asarray(out, dtype=np.int16)

    def _exact_one(self, A) -> bool:
        return abs(det_bareiss(np.asarray(A, dtype=np.int64).tolist())) == 1

    def _random_full_rank(self, R, rng, attempts=96):
        n = len(R)
        for _ in range(attempts):
            idx = rng.sample(range(n), self.d)
            A = R[idx].astype(np.int64)
            d0 = int(round(float(np.linalg.det(A.astype(np.float64)))))
            if d0:
                return A, d0
        chosen = []
        rank = 0
        for v in R:
            B = np.asarray(chosen + [v.tolist()], dtype=np.float64)
            rr = int(np.linalg.matrix_rank(B, tol=1e-8))
            if rr > rank:
                chosen.append(v.tolist())
                rank = rr
                if rank == self.d:
                    A = np.asarray(chosen, dtype=np.int64)
                    return A, int(round(float(np.linalg.det(A.astype(np.float64)))))
        return None, 0

    def _heuristic_basis(self, E: np.ndarray, seed: int):
        R = self._sign_reps(E)
        if len(R) < self.d:
            return None
        rng = random.Random(seed)
        n = len(R)
        for _ in range(8):
            inds = np.asarray(
                [[rng.randrange(n) for _ in range(self.d)] for __ in range(256)],
                dtype=np.int64,
            )
            B = R[inds].astype(np.float64)
            ds = np.rint(np.linalg.det(B)).astype(np.int64)
            for h in np.flatnonzero(np.abs(ds) == 1)[:8]:
                A = R[inds[int(h)]].astype(np.int64)
                if self._exact_one(A):
                    return A
        for _ in range(24):
            A, d0 = self._random_full_rank(R, rng)
            if A is None:
                return None
            if abs(d0) == 1 and self._exact_one(A):
                return A
            seen = set()
            for step in range(48):
                d0 = int(round(float(np.linalg.det(A.astype(np.float64)))))
                if d0 == 0:
                    break
                key = tuple(map(tuple, A.tolist()))
                if key in seen:
                    break
                seen.add(key)
                try:
                    adjcols = np.rint(
                        d0 * np.linalg.inv(A.astype(np.float64))
                    ).astype(np.int64)
                except np.linalg.LinAlgError:
                    break
                best = None
                order = list(range(self.d))
                rng.shuffle(order)
                for i in order:
                    c = adjcols[:, i]
                    vals = R.astype(np.int64) @ c
                    for h in np.flatnonzero(np.abs(vals) == 1)[:8]:
                        B = A.copy()
                        B[i] = R[int(h)]
                        if self._exact_one(B):
                            return B
                    nz = np.flatnonzero(vals)
                    if len(nz):
                        av = np.abs(vals[nz])
                        j = int(nz[int(np.argmin(av))])
                        score = int(abs(vals[j]))
                        if best is None or score < best[0]:
                            best = (score, i, j)
                if best is None:
                    break
                score, i, j = best
                if score >= abs(d0) and step > 3:
                    break
                A[i] = R[j]
                if score == 1 and self._exact_one(A):
                    return A
        return None

    def find_basis(self, emask: int, seed: int):
        A, method = self._catalogue_lookup(emask)
        if A is not None:
            if not self._exact_one(A):
                raise AssertionError("non-unimodular catalogue hit")
            return A, method
        E = self.mask_to_points(emask)
        A = self._heuristic_basis(E, seed)
        if A is None:
            self.stats["heuristic_failure"] += 1
            return None, "failure"
        if not self._exact_one(A):
            raise AssertionError("heuristic accepted non-unimodular matrix")
        self._add_basis(A)
        bmask = self._basis_mask(A)
        self.emask_basis[emask] = (bmask, A.copy())
        self.stats["heuristic"] += 1
        return A, "heuristic"


def connect():
    client = MongoClient(
        URI,
        tls=True,
        directConnection=True,
        serverSelectionTimeoutMS=30000,
        connectTimeoutMS=30000,
        socketTimeoutMS=1800000,
    )
    client.admin.command("ping")
    return client, client.polydb[COLLECTION]


def outlier_query_test(coll):
    known_outlier = {"_id": "F.9D.f11.0000000", **OUTLIER}
    known_standard = {"_id": "F.9D.f18.0666150", **OUTLIER}
    a = coll.count_documents(known_outlier, maxTimeMS=60000)
    b = coll.count_documents(known_standard, maxTimeMS=60000)
    if (a, b) != (1, 0):
        raise AssertionError(f"nested outlier query failed self-test: {(a,b)}")


def d9_ranges(shard: int, nshards: int):
    total = sum(D9_STRATA.values())
    lo = total * shard // nshards
    hi = total * (shard + 1) // nshards
    offset = 0
    out = []
    for f, count in sorted(D9_STRATA.items()):
        a = max(lo, offset)
        b = min(hi, offset + count)
        if a < b:
            local_a = a - offset
            local_b = b - offset
            out.append((f, local_a, local_b))
        offset += count
    return lo, hi, out


def id_seed(poly_id: str) -> int:
    return int.from_bytes(hashlib.sha256(poly_id.encode()).digest()[:8], "big")


def scan_d9(args):
    client, coll = connect()
    outlier_query_test(coll)
    engine = EwaldEngine(9, seed_catalogue=args.seed_catalogue)
    glo, ghi, ranges = d9_ranges(args.shard, args.nshards)
    assigned = ghi - glo
    processed = 0
    outliers = 0
    failures = []
    anomalies = []
    min_E = engine.npoints + 1
    min_ids = []
    methods = Counter()
    start = time.time()
    projection = {"_id": 1, "FACETS": 1, "N_FACETS": 1, "SMOOTH": 1, "REFLEXIVE": 1}

    for f, a, b in ranges:
        p = f"F.9D.f{f}."
        q = {
            "_id": {"$gte": p + f"{a:07d}", "$lt": p + f"{b:07d}"},
            **OUTLIER,
        }
        cursor = coll.find(q, projection=projection, batch_size=1000).sort("_id", 1)
        if args.limit:
            cursor = cursor.limit(max(0, args.limit - processed))
        segment = 0
        for doc in cursor:
            processed += 1
            segment += 1
            outliers += 1
            pid = doc["_id"]
            U = parse_facets(doc["FACETS"], 9)
            expected = -np.eye(9, dtype=np.int16)
            if len(U) < 9 or not np.array_equal(U[:9], expected):
                anomalies.append({"id": pid, "facets": U.astype(int).tolist()})
                print("NORMAL_FORM_ANOMALY", pid, flush=True)
                continue
            if all(abs(int(x)) <= 1 for x in U.ravel()):
                raise AssertionError(f"outlier query false positive for {pid}")
            emask = engine.ewald_mask(U)
            ne = emask.bit_count()
            if ne < min_E:
                min_E = ne
                min_ids = [pid]
            elif ne == min_E:
                min_ids.append(pid)
            basis, method = engine.find_basis(emask, id_seed(pid))
            methods[method] += 1
            if basis is None:
                E = engine.mask_to_points(emask)
                failure = {
                    "id": pid,
                    "n_facets": int(len(U)),
                    "n_E": int(ne),
                    "rank2": rank_mod(E, 2),
                    "rank3": rank_mod(E, 3),
                    "rank5": rank_mod(E, 5),
                    "facets": U.astype(int).tolist(),
                    "E": E.astype(int).tolist(),
                }
                failures.append(failure)
                print("HEURISTIC_FAILURE", json.dumps({k:v for k,v in failure.items() if k not in ("facets","E")}), flush=True)
            else:
                d = det_bareiss(basis.tolist())
                if abs(d) != 1:
                    raise AssertionError("inexact basis accepted")
            if processed % 10000 == 0:
                print(
                    "PROGRESS",
                    args.shard,
                    processed,
                    "outliers",
                    outliers,
                    "minE",
                    min_E,
                    "failures",
                    len(failures),
                    "catalogue",
                    len(engine.catalogue),
                    "normal_masks",
                    len(engine.normal_masks),
                    "elapsed",
                    round(time.time() - start, 1),
                    flush=True,
                )
            if args.limit and processed >= args.limit:
                break
        print("SEGMENT", f, a, b, "outliers", segment, flush=True)
        if args.limit and processed >= args.limit:
            break

    standard = assigned - outliers if not args.limit else None
    result = {
        "dimension": 9,
        "mode": "outliers",
        "shard": args.shard,
        "nshards": args.nshards,
        "global_start": glo,
        "global_end": ghi,
        "assigned_total": assigned,
        "processed_outliers": outliers,
        "standard_basis_records": standard,
        "query_self_test": True,
        "min_E_outliers": None if min_E > engine.npoints else min_E,
        "min_ids": min_ids,
        "methods": dict(methods),
        "failures": failures,
        "normal_form_anomalies": anomalies,
        "unique_E_masks": len(engine.emask_basis),
        "unique_normal_masks": len(engine.normal_masks),
        "basis_catalogue_size": len(engine.catalogue),
        "elapsed_sec": time.time() - start,
        "limited": bool(args.limit),
    }
    client.close()
    return result


def scan_d7(args):
    client, coll = connect()
    engine = EwaldEngine(7, seed_catalogue=args.seed_catalogue)
    q = {"_id": {"$gte": "F.7D.", "$lt": "F.7D/"}}
    projection = {"_id": 1, "FACETS": 1, "N_FACETS": 1, "SMOOTH": 1, "REFLEXIVE": 1}
    cursor = coll.find(q, projection=projection, batch_size=1000).sort("_id", 1)
    if args.limit:
        cursor = cursor.limit(args.limit)
    count = 0
    failures = []
    anomalies = []
    min_E = engine.npoints + 1
    min_ids = []
    methods = Counter()
    start = time.time()
    for doc in cursor:
        count += 1
        pid = doc["_id"]
        U = parse_facets(doc["FACETS"], 7)
        if len(U) < 7 or not np.array_equal(U[:7], -np.eye(7, dtype=np.int16)):
            anomalies.append({"id": pid, "facets": U.astype(int).tolist()})
            continue
        emask = engine.ewald_mask(U)
        ne = emask.bit_count()
        if ne < min_E:
            min_E = ne
            min_ids = [pid]
        elif ne == min_E:
            min_ids.append(pid)
        basis, method = engine.find_basis(emask, id_seed(pid))
        methods[method] += 1
        if basis is None:
            E = engine.mask_to_points(emask)
            failures.append({
                "id": pid,
                "n_E": ne,
                "rank2": rank_mod(E, 2),
                "rank3": rank_mod(E, 3),
                "facets": U.astype(int).tolist(),
                "E": E.astype(int).tolist(),
            })
        if count % 10000 == 0:
            print("D7_PROGRESS", count, "minE", min_E, "failures", len(failures), "elapsed", round(time.time()-start,1), flush=True)
    result = {
        "dimension": 7,
        "mode": "full",
        "count": count,
        "min_E": None if min_E > engine.npoints else min_E,
        "min_ids": min_ids,
        "methods": dict(methods),
        "failures": failures,
        "normal_form_anomalies": anomalies,
        "unique_E_masks": len(engine.emask_basis),
        "unique_normal_masks": len(engine.normal_masks),
        "basis_catalogue_size": len(engine.catalogue),
        "elapsed_sec": time.time() - start,
        "limited": bool(args.limit),
    }
    client.close()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dimension", type=int, choices=(7, 9), required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed-catalogue", type=int, default=768)
    ap.add_argument("--outdir", default="ewald-polydb-results")
    args = ap.parse_args()
    if not (0 <= args.shard < args.nshards):
        ap.error("invalid shard")
    os.makedirs(args.outdir, exist_ok=True)
    result = scan_d7(args) if args.dimension == 7 else scan_d9(args)
    path = os.path.join(args.outdir, f"d{args.dimension}-shard-{args.shard}.json")
    with open(path, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    compact = {k:v for k,v in result.items() if k not in ("failures","normal_form_anomalies")}
    compact["n_failures"] = len(result["failures"])
    compact["n_normal_form_anomalies"] = len(result["normal_form_anomalies"])
    print("RESULT", json.dumps(compact, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
