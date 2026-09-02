#!/usr/bin/env python3
from __future__ import annotations
import itertools, json, math
import numpy as np
from pymongo import MongoClient, ASCENDING, DESCENDING

URI = 'mongodb://polymake:database@db.polymake.org:27017'
COLLECTION = 'Polytopes.Lattice.SmoothReflexive'
D = 9
client = MongoClient(URI, tls=True, directConnection=True,
                     serverSelectionTimeoutMS=30000,
                     connectTimeoutMS=30000,
                     socketTimeoutMS=180000)
client.admin.command('ping')
coll = client.polydb[COLLECTION]
print('PING_OK', flush=True)


def bounds(f):
    p = f'F.9D.f{f}.'
    return {'_id': {'$gte': p, '$lt': p[:-1] + '/'}}


def mat(v):
    rows = []
    for r in v:
        if not isinstance(r, list):
            continue
        rows.append([int(x) for x in r])
    return np.asarray(rows, dtype=np.int64)


def det_bareiss(A):
    a = [[int(x) for x in row] for row in A]
    n = len(a); sign = 1; prev = 1
    for k in range(n-1):
        if a[k][k] == 0:
            p = next((i for i in range(k+1,n) if a[i][k]), None)
            if p is None: return 0
            a[k],a[p] = a[p],a[k]; sign = -sign
        pivot = a[k][k]
        for i in range(k+1,n):
            for j in range(k+1,n):
                a[i][j] = (a[i][j]*pivot-a[i][k]*a[k][j])//prev
        prev = pivot
        for i in range(k+1,n): a[i][k] = 0
        for j in range(k+1,n): a[k][j] = 0
    return sign*a[-1][-1]


def inspect(doc):
    F = mat(doc['FACETS'])
    const = F[:,0]; U = F[:,1:]
    first_det = det_bareiss(U[:D]) if len(U) >= D else 0
    identity = bool(len(U)>=D and np.array_equal(U[:D], np.eye(D,dtype=np.int64)))
    primitive = all(math.gcd(*map(abs,row.tolist())) == 1 for row in U)
    # Find incident facets at vertex 0 from facet-to-vertex incidence.
    vif = [r for r in doc.get('VERTICES_IN_FACETS',[]) if isinstance(r,list)]
    incident0 = [i for i,r in enumerate(vif) if 0 in r]
    inc_det = None
    if len(incident0) == D:
        inc_det = det_bareiss(U[incident0])
    return {
        'id': doc['_id'], 'n_facets': len(U), 'n_vertices': doc.get('N_VERTICES'),
        'constants': sorted(set(map(int,const.tolist()))),
        'primitive': primitive, 'first9_det': first_det, 'first9_identity': identity,
        'incident0': incident0, 'incident0_det': inc_det,
        'normal_min': int(U.min()), 'normal_max': int(U.max()),
        'facets': F.tolist(),
    }

projection = {'_id':1,'FACETS':1,'N_FACETS':1,'N_VERTICES':1,'VERTICES_IN_FACETS':1}
total = 0
summary = []
for f in range(10, 40):
    q = bounds(f)
    c = coll.count_documents(q, maxTimeMS=180000)
    if not c:
        continue
    total += c
    first = coll.find_one(q, projection, sort=[('_id',ASCENDING)], max_time_ms=60000)
    last = coll.find_one(q, projection, sort=[('_id',DESCENDING)], max_time_ms=60000)
    sample_indices = sorted(set([0, 1 if c>1 else 0, c//4, c//2, (3*c)//4, c-1]))
    samples = []
    for i in sample_indices:
        sid = f'F.9D.f{f}.{i:07d}'
        d = coll.find_one({'_id':sid}, projection, max_time_ms=60000)
        if d is None:
            # Fall back to indexed skip for unexpected numbering.
            d = coll.find_one(q, projection, skip=i, sort=[('_id',ASCENDING)], max_time_ms=180000)
        samples.append(inspect(d))
    item = {'f':f,'count':c,'first_id':first['_id'],'last_id':last['_id'],'samples':samples}
    summary.append(item)
    print('STRATUM', json.dumps(item,separators=(',',':')), flush=True)
print('TOTAL', total, flush=True)
print('SUMMARY', json.dumps(summary,separators=(',',':')), flush=True)
