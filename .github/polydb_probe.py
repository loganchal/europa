#!/usr/bin/env python3
import json
from pymongo import MongoClient

client = MongoClient(
    'mongodb://polymake:database@db.polymake.org:27017',
    tls=True,
    directConnection=True,
    serverSelectionTimeoutMS=30000,
)
client.admin.command('ping')
db = client.polydb
names = sorted(n for n in db.list_collection_names() if 'SmoothReflexive' in n)
print('COLLECTIONS', json.dumps(names))
for name in names:
    coll = db[name]
    print('COLLECTION', name, 'COUNT', coll.count_documents({}))
    for filt in ({'CONE_DIM': 10}, {'DIM': 9}, {'CONE_DIM': 9}, {}):
        doc = coll.find_one(filt)
        if doc is None:
            print('FILTER', json.dumps(filt), 'NONE')
            continue
        def summary(v):
            if isinstance(v, dict):
                return {'type':'dict','keys':sorted(v.keys())[:30]}
            if isinstance(v, list):
                out={'type':'list','len':len(v)}
                if v: out['first']=summary(v[0])
                return out
            return {'type':type(v).__name__,'value':str(v)[:200]}
        print('FILTER', json.dumps(filt), 'ID', doc.get('_id'))
        print('KEYS', json.dumps(sorted(doc.keys())))
        print('SUMMARY', json.dumps({k:summary(v) for k,v in doc.items()}, default=str))
        break
