#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import json
import time

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
    U = np.asarray(rows, dtype=np.int64)
    if U.ndim != 2 or U.shape[1] != d:
        raise ValueError("malformed facet matrix")
    return U


def exact_unimodular_inverse(B: np.ndarray) -> np.ndarray:
    B = np.asarray(B, dtype=np.int64)
    d = len(B)
    determinant = scan.det_bareiss(B.tolist())
    if abs(determinant) != 1:
        raise ValueError(f"normalizing matrix determinant is {determinant}")
    candidate = np.rint(np.linalg.inv(B.astype(np.float64))).astype(np.int64)
    if np.array_equal(B @ candidate, np.eye(d, dtype=np.int64)):
        return candidate
    aug = [
        [Fraction(int(B[i, j])) for j in range(d)]
        + [Fraction(int(i == j)) for j in range(d)]
        for i in range(d)
    ]
    for col in range(d):
        pivot = next((r for r in range(col, d) if aug[r][col]), None)
        if pivot is None:
            raise ValueError("singular normalizing matrix")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        q = aug[col][col]
        aug[col] = [x / q for x in aug[col]]
        for r in range(d):
            if r == col:
                continue
            q = aug[r][col]
            if q:
                aug[r] = [x - q * y for x, y in zip(aug[r], aug[col])]
    right = [row[d:] for row in aug]
    if any(x.denominator != 1 for row in right for x in row):
        raise ValueError("unimodular inverse was not integral")
    inv = np.asarray([[int(x) for x in row] for row in right], dtype=np.int64)
    if not np.array_equal(B @ inv, np.eye(d, dtype=np.int64)):
        raise AssertionError("exact inverse verification failed")
    return inv


def normalized_coordinates(U: np.ndarray, d: int):
    if len(U) < d:
        raise ValueError("too few facets")
    B = U[:d]
    Binv = exact_unimodular_inverse(B)
    C = U @ Binv
    if not np.array_equal(C[:d], np.eye(d, dtype=np.int64)):
        raise AssertionError("normal coordinate transformation failed")
    return C, B, Binv


def verify_original_basis(U: np.ndarray, Binv: np.ndarray, A: np.ndarray):
    X = np.asarray(A, dtype=np.int64) @ Binv.T
    determinant = scan.det_bareiss(X.tolist())
    if abs(determinant) != 1:
        raise AssertionError(f"mapped basis determinant is {determinant}")
    if np.max(np.abs(U @ X.T)) > 1:
        raise AssertionError("mapped basis is not in the symmetric point set")
    return X


def failure_record(pid, U, C, Binv, engine, emask, include_rank5=False):
    Ey = engine.mask_to_points(emask).astype(np.int64)
    Ex = Ey @ Binv.T
    item = {
        "id": pid,
        "n_facets": int(len(U)),
        "n_E": int(len(Ex)),
        "rank2": scan.rank_mod(Ex, 2),
        "rank3": scan.rank_mod(Ex, 3),
        "primitive_facets": U.astype(int).tolist(),
        "normal_form_facets": C.astype(int).tolist(),
        "normalizing_inverse": Binv.astype(int).tolist(),
        "E": Ex.astype(int).tolist(),
    }
    if include_rank5:
        item["rank5"] = scan.rank_mod(Ex, 5)
    return item


def scan_d7(args):
    client, coll = scan.connect()
    engine = scan.EwaldEngine(7, seed_catalogue=args.seed_catalogue)
    q = {"_id": {"$gte": "F.7D.", "$lt": "F.7D/"}}
    projection = {"_id": 1, "FACETS": 1, "N_FACETS": 1}
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
        U = parse_primitive_facets(doc["FACETS"], 7)
        try:
            C, B, Binv = normalized_coordinates(U, 7)
        except Exception as exc:
            anomalies.append({"id": pid, "error": str(exc), "primitive_facets": U.astype(int).tolist()})
            continue
        emask = engine.ewald_mask(C)
        ne = emask.bit_count()
        if ne < min_E:
            min_E = ne
            min_ids = [pid]
        elif ne == min_E:
            min_ids.append(pid)
        basis, method = engine.find_basis(emask, scan.id_seed(pid))
        methods[method] += 1
        if basis is None:
            failures.append(failure_record(pid, U, C, Binv, engine, emask))
            print("D7_HEURISTIC_FAILURE", pid, "E", ne, flush=True)
        else:
            verify_original_basis(U, Binv, basis)
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


def scan_d9(args):
    client, coll = scan.connect()
    scan.outlier_query_test(coll)
    engine = scan.EwaldEngine(9, seed_catalogue=args.seed_catalogue)
    glo, ghi, ranges = scan.d9_ranges(args.shard, args.nshards)
    assigned = ghi - glo
    query_hits = 0
    failures = []
    anomalies = []
    min_E = engine.npoints + 1
    min_ids = []
    methods = Counter()
    start = time.time()
    projection = {"_id": 1, "FACETS": 1, "N_FACETS": 1}

    for f, a, b in ranges:
        p = f"F.9D.f{f}."
        q = {
            "_id": {"$gte": p + f"{a:07d}", "$lt": p + f"{b:07d}"},
            **scan.OUTLIER,
        }
        cursor = coll.find(q, projection=projection, batch_size=1000).sort("_id", 1)
        if args.limit:
            cursor = cursor.limit(max(0, args.limit - query_hits))
        segment = 0
        for doc in cursor:
            query_hits += 1
            segment += 1
            pid = doc["_id"]
            U = parse_primitive_facets(doc["FACETS"], 9)
            try:
                C, B, Binv = normalized_coordinates(U, 9)
            except Exception as exc:
                anomalies.append({"id": pid, "error": str(exc), "primitive_facets": U.astype(int).tolist()})
                print("NORMAL_FORM_ANOMALY", pid, str(exc), flush=True)
                continue
            emask = engine.ewald_mask(C)
            ne = emask.bit_count()
            if ne < min_E:
                min_E = ne
                min_ids = [pid]
            elif ne == min_E:
                min_ids.append(pid)
            basis, method = engine.find_basis(emask, scan.id_seed(pid))
            methods[method] += 1
            if basis is None:
                item = failure_record(pid, U, C, Binv, engine, emask, include_rank5=True)
                failures.append(item)
                print("HEURISTIC_FAILURE", json.dumps({k:v for k,v in item.items() if k not in ("primitive_facets","normal_form_facets","normalizing_inverse","E")}), flush=True)
            else:
                verify_original_basis(U, Binv, basis)
            if query_hits % 10000 == 0:
                print(
                    "PROGRESS", args.shard, query_hits, "query_hits", query_hits,
                    "minE", min_E, "failures", len(failures),
                    "catalogue", len(engine.catalogue),
                    "normal_masks", len(engine.normal_masks),
                    "elapsed", round(time.time()-start,1), flush=True,
                )
            if args.limit and query_hits >= args.limit:
                break
        print("SEGMENT", f, a, b, "query_hits", segment, flush=True)
        if args.limit and query_hits >= args.limit:
            break

    result = {
        "dimension": 9,
        "mode": "raw-coefficient-query-superset",
        "shard": args.shard,
        "nshards": args.nshards,
        "global_start": glo,
        "global_end": ghi,
        "assigned_total": assigned,
        "processed_query_hits": query_hits,
        "raw_trivial_standard_basis_records": None if args.limit else assigned - query_hits,
        "query_self_test": True,
        "min_E_query_hits": None if min_E > engine.npoints else min_E,
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


scan.parse_facets = parse_primitive_facets
scan.scan_d7 = scan_d7
scan.scan_d9 = scan_d9
scan.main()
