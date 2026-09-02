#!/usr/bin/env python3
from __future__ import annotations
import json
from bson import json_util
from pymongo import MongoClient

URI = 'mongodb://polymake:database@db.polymake.org:27017'
COLLECTION = 'Polytopes.Lattice.SmoothReflexive'
client = MongoClient(URI, tls=True, directConnection=True,
                     serverSelectionTimeoutMS=30000,
                     connectTimeoutMS=30000,
                     socketTimeoutMS=60000)
client.admin.command('ping')
print('PING_OK', flush=True)
db = client.polydb
coll = db[COLLECTION]

queries = [
    {},
    {'_id': {'$gte': 'F.9D.', '$lt': 'F.9D/'}},
    {'_id': {'$regex': r'^F\.9D\.'}},
    {'CONE_DIM': 10},
    {'DIM': 9},
]

def shape(v):
    if isinstance(v, dict):
        out = {'type': 'dict', 'keys': sorted(map(str, v.keys()))}
        for k in ('data','rows','cols','dim','type','value'):
            if k in v:
                out[k] = shape(v[k])
        return out
    if isinstance(v, (list, tuple)):
        out = {'type': type(v).__name__, 'len': len(v)}
        if v:
            out['first'] = shape(v[0])
            if len(v) > 1:
                out['second'] = shape(v[1])
        return out
    return {'type': type(v).__name__, 'repr': repr(v)[:500]}

for q in queries:
    try:
        doc = coll.find_one(q, max_time_ms=30000)
        print('QUERY', json.dumps(q, default=str), 'FOUND', doc is not None, flush=True)
        if doc is None:
            continue
        print('ID', doc.get('_id'), flush=True)
        print('KEYS', json.dumps(sorted(doc.keys())), flush=True)
        print('SHAPES', json.dumps({k: shape(v) for k,v in doc.items()}, default=str), flush=True)
        print('DOCUMENT', json_util.dumps(doc)[:50000], flush=True)
        if q:
            break
    except Exception as exc:
        print('QUERY_ERROR', json.dumps(q, default=str), type(exc).__name__, str(exc), flush=True)

for info_name in ('_collectionInfo.' + COLLECTION,
                  '_sectionInfo.Polytopes.Lattice'):
    try:
        d = db[info_name].find_one({}, max_time_ms=30000)
        print('INFO', info_name, 'FOUND', d is not None, flush=True)
        if d:
            print('INFO_KEYS', json.dumps(sorted(d.keys())), flush=True)
            print('INFO_DOC', json_util.dumps(d)[:50000], flush=True)
    except Exception as exc:
        print('INFO_ERROR', info_name, type(exc).__name__, str(exc), flush=True)
