import argparse,gzip,json,math,urllib.request,collections,time
D=9
URL='https://polymake.org/polytopes/paffenholz/data/smooth-fano/d9/fv-09-{}p.gz'

def polytopes(path):
    cur=[]; inpoly=False
    with gzip.open(path,'rt') as f:
        for raw in f:
            s=raw.strip()
            if not inpoly:
                if not s: continue
                if s!='FACETS': raise ValueError(s)
                inpoly=True; cur=[]; continue
            if not s:
                yield cur; inpoly=False; continue
            v=[int(x) for x in s.split()]
            if len(v)!=10 or v[0]!=1: raise ValueError(s)
            cur.append(v[1:])
    if inpoly: yield cur

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--facets',type=int,required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    path=f'/tmp/fv-09-{a.facets}p.gz';urllib.request.urlretrieve(URL.format(a.facets),path)
    hist=collections.Counter(); movable=[]; maxbox=(0,None); scanned=0;t=time.time()
    std=[[-1 if i==j else 0 for i in range(D)] for j in range(D)]
    for idx,rays in enumerate(polytopes(path),1):
        scanned+=1
        if rays[:D]!=std: raise ValueError(('not standard',idx,rays[:D]))
        sums=[sum(v) for v in rays[D:]]
        if any(s<0 for s in sums): raise ValueError(('negative slack',idx,sums))
        box=math.prod(2*s+1 for s in sums)
        nz=sum(s>0 for s in sums)
        hist[(nz,box)]+=1
        if box>1:
            rec={'ordinal':idx,'box':box,'sums':sums,'rays':rays}
            movable.append(rec)
            if box>maxbox[0]:maxbox=(box,idx)
        if scanned%500000==0: print(json.dumps({'facets':a.facets,'scanned':scanned,'movable':len(movable),'maxbox':maxbox}),flush=True)
    out={'facets':a.facets,'scanned':scanned,'movable_count':len(movable),'max_box':maxbox[0],'max_box_ordinal':maxbox[1],
         'hist':[{'nonzero_vars':k[0],'box':k[1],'count':v} for k,v in sorted(hist.items())],
         'movable':movable,'elapsed':time.time()-t}
    with open(a.out,'w') as f:json.dump(out,f,separators=(',',':'))
    print('FINAL '+json.dumps({k:v for k,v in out.items() if k not in ('movable','hist')}),flush=True)
if __name__=='__main__':main()
