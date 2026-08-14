import gzip,json,tempfile,urllib.request
from neat_filter9 import records,URL
TARGET=2822308
FACETS=16

tmp=tempfile.NamedTemporaryFile(suffix='.gz',delete=False);tmp.close()
urllib.request.urlretrieve(URL.format(FACETS),tmp.name)
for idx,U in enumerate(records(tmp.name),start=1):
    if idx==TARGET:
        out={'dimension':9,'facets':FACETS,'ordinal':TARGET,'normals':U.tolist(),'displacement':[0]*9+[-1,-1,-1,-1,-1,-1,-3]}
        open('seed9-16-2822308.json','w').write(json.dumps(out,separators=(',',':')))
        print(json.dumps(out))
        break
else:
    raise SystemExit('target not found')
