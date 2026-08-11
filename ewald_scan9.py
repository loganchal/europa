import argparse, gzip, itertools, json, random, time, urllib.request
import numpy as np

D=9
POINTS=np.array(list(itertools.product((-1,0,1), repeat=D)),dtype=np.int16)
PTUP=[tuple(map(int,x)) for x in POINTS]
NPTS=len(POINTS)
FULL=(1<<NPTS)-1
INDEX={x:i for i,x in enumerate(PTUP)}
BASE='https://polymake.org/polytopes/paffenholz/data/smooth-fano/d9/fv-09-{}p.gz'
RAY_MASK={}

def ray_mask(ray):
    ray=tuple(ray)
    got=RAY_MASK.get(ray)
    if got is not None:return got
    ok=np.abs(POINTS@np.asarray(ray,dtype=np.int16))<=1
    got=int.from_bytes(np.packbits(ok,bitorder='little').tobytes(),'little') & FULL
    RAY_MASK[ray]=got
    return got

def ewald_mask(facets):
    m=FULL
    for a in facets:m &= ray_mask(a)
    return m

def indices(m):
    out=[]
    while m:
        b=m&-m;out.append(b.bit_length()-1);m^=b
    return out

def det_bareiss(ids):
    A=[list(PTUP[i]) for i in ids];n=D;sg=1;prev=1
    for k in range(n-1):
        if A[k][k]==0:
            q=next((i for i in range(k+1,n) if A[i][k]),None)
            if q is None:return 0
            A[k],A[q]=A[q],A[k];sg=-sg
        piv=A[k][k]
        for i in range(k+1,n):
            aik=A[i][k]
            for j in range(k+1,n):A[i][j]=(A[i][j]*piv-aik*A[k][j])//prev
        prev=piv
        for i in range(k+1,n):A[i][k]=0
    return sg*A[-1][-1]

def basis_mask(ids):
    m=0
    for i in ids:m|=1<<i
    return m

def sign_reps(ids):
    out=[]
    for i in ids:
        for z in PTUP[i]:
            if z:
                if z>0:out.append(i)
                break
    return out

def find_basis(m,library,seed,trials=900):
    for bm in library:
        if m & bm == bm:return bm
    ids=indices(m);reps=sign_reps(ids)
    if len(reps)<D:return None
    low=sorted(reps,key=lambda i:(sum(abs(z) for z in PTUP[i]),i))
    pools=(low[:min(96,len(low))],low[:min(320,len(low))],reps)
    rng=random.Random(seed);per=max(1,trials//len(pools))
    for pool in pools:
        if len(pool)<D:continue
        for _ in range(per):
            sel=rng.sample(pool,D)
            if abs(det_bareiss(sel))==1:return basis_mask(sel)
    return None

def rank_mod(ids,p):
    A=[[z%p for z in PTUP[i]] for i in ids if any(PTUP[i])];r=0
    for c in range(D):
        q=next((i for i in range(r,len(A)) if A[i][c]),None)
        if q is None:continue
        A[r],A[q]=A[q],A[r]
        inv=pow(A[r][c],-1,p);A[r]=[(z*inv)%p for z in A[r]]
        for i in range(r+1,len(A)):
            if A[i][c]:
                f=A[i][c];A[i]=[(A[i][j]-f*A[r][j])%p for j in range(D)]
        r+=1
        if r==D:break
    return r

def parse(url,expected):
    with urllib.request.urlopen(url,timeout=120) as raw:
        with gzip.GzipFile(fileobj=raw) as gz:
            cur=[]
            for bb in gz:
                s=bb.decode('ascii').strip()
                if not s:continue
                if s=='FACETS':
                    if cur:
                        if len(cur)!=expected:raise ValueError(('facet count',len(cur),expected))
                        yield cur;cur=[]
                    continue
                q=[int(z) for z in s.split()]
                if len(q)!=D+1 or q[0]!=1:raise ValueError(('bad row',q))
                cur.append(tuple(q[1:]))
            if cur:
                if len(cur)!=expected:raise ValueError(('facet count',len(cur),expected))
                yield cur

def record(facets,ec,idx,extra=None):
    d={'index_in_file':idx,'ewald_count':ec,'facets':[list(a) for a in facets]}
    if extra:d.update(extra)
    return d

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--facets',type=int,required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    k=a.facets;url=BASE.format(k);t0=time.time();scanned=0;alarms=[];best=[]
    std=[]
    for j in range(D):
        e=[0]*D;e[j]=1;std.append(INDEX[tuple(e)])
    library=[basis_mask(std)]
    for idx,facets in enumerate(parse(url,k),1):
        scanned+=1;m=ewald_mask(facets);ec=m.bit_count()
        if len(best)<20 or ec<best[-1]['ewald_count']:
            best.append(record(facets,ec,idx));best.sort(key=lambda z:z['ewald_count']);best=best[:20]
        bm=find_basis(m,library,(k<<32)^idx)
        if bm is None:
            ids=indices(m);extra={'ranks':{str(p):rank_mod(ids,p) for p in (2,3,5,7,11)}}
            alarms.append(record(facets,ec,idx,extra))
            if len(alarms)<=10:print('ALARM',json.dumps(alarms[-1],separators=(',',':')),flush=True)
        elif bm not in library and len(library)<768:library.append(bm)
        if scanned%100000==0:
            print(json.dumps({'facets':k,'scanned':scanned,'bestE':best[0]['ewald_count'],'alarms':len(alarms),'library':len(library),'ray_cache':len(RAY_MASK),'sec':round(time.time()-t0,1)}),flush=True)
    out={'dimension':D,'facet_count':k,'url':url,'scanned':scanned,'elapsed_sec':time.time()-t0,'alarm_count':len(alarms),'alarms':alarms,'best':best,'basis_library_size':len(library),'ray_cache_size':len(RAY_MASK)}
    with open(a.out,'w') as f:json.dump(out,f,separators=(',',':'))
    print('FINAL '+json.dumps({x:y for x,y in out.items() if x not in ('alarms','best')},separators=(',',':'))+f' bestE={best[0]["ewald_count"] if best else None}',flush=True)
if __name__=='__main__':main()
