import argparse, itertools, json, random, time, urllib.request
import numpy as np

D=8
POINTS=np.array(list(itertools.product((-1,0,1), repeat=D)),dtype=np.int16)
NPTS=len(POINTS)
FULL=(1<<NPTS)-1
INDEX={tuple(map(int,x)):i for i,x in enumerate(POINTS)}
RAW='https://raw.githubusercontent.com/GorrieXIV/Magma/master/libs/data/polytopes/smoothfano8/block{}'
RAY_MASK={}

def digits(n,b):
    out=[]
    while n:
        out.append(n%b); n//=b
    return out

def decode(line,base):
    a=digits(int(line),base)
    if len(a)<2 or a[0]!=D:
        raise ValueError(('bad dimension',a[:4]))
    shift=a[1]
    c=[z-shift for z in a[2:]]
    if len(c)%D:
        raise ValueError(('bad packed length',len(c)))
    return [tuple(c[i:i+D]) for i in range(0,len(c),D)]

def ray_mask(ray):
    ray=tuple(ray)
    got=RAY_MASK.get(ray)
    if got is not None: return got
    ok=np.abs(POINTS@np.array(ray,dtype=np.int16))<=1
    got=int.from_bytes(np.packbits(ok,bitorder='little').tobytes(),'little') & FULL
    RAY_MASK[ray]=got
    return got

def ewald_mask(rays):
    m=FULL
    for r in rays:
        m &= ray_mask(r)
    return m

def det_bareiss(rows):
    A=[list(map(int,row)) for row in rows]
    n=len(A); sign=1; prev=1
    for k in range(n-1):
        if A[k][k]==0:
            q=next((i for i in range(k+1,n) if A[i][k]),None)
            if q is None:return 0
            A[k],A[q]=A[q],A[k]; sign=-sign
        piv=A[k][k]
        for i in range(k+1,n):
            for j in range(k+1,n):
                A[i][j]=(A[i][j]*piv-A[i][k]*A[k][j])//prev
        prev=piv
        for i in range(k+1,n):A[i][k]=0
    return sign*A[-1][-1]

def indices(m):
    out=[]
    while m:
        b=m&-m; out.append(b.bit_length()-1); m^=b
    return out

def rank_mod(ids,p):
    A=[[int(x)%p for x in POINTS[i]] for i in ids if np.any(POINTS[i])]
    r=0
    for c in range(D):
        q=next((i for i in range(r,len(A)) if A[i][c]%p),None)
        if q is None:continue
        A[r],A[q]=A[q],A[r]
        inv=pow(A[r][c],-1,p)
        A[r]=[(x*inv)%p for x in A[r]]
        for i in range(r+1,len(A)):
            if A[i][c]%p:
                f=A[i][c]%p
                A[i]=[(A[i][j]-f*A[r][j])%p for j in range(D)]
        r+=1
        if r==D:break
    return r

def basis_mask(sel):
    m=0
    for i in sel:m|=1<<i
    return m

def find_basis(m,library,rng,trials=1200):
    for bm in library:
        if m & bm == bm:return bm
    ids=indices(m)
    low=sorted(ids,key=lambda i:(int(np.abs(POINTS[i]).sum()),i))
    pools=(low[:min(128,len(low))],low[:min(512,len(low))],ids)
    per=max(1,trials//len(pools))
    for pool in pools:
        if len(pool)<D:continue
        for _ in range(per):
            sel=rng.sample(pool,D)
            if abs(det_bareiss([POINTS[i] for i in sel]))==1:
                return basis_mask(sel)
    return None

def ewald_points(ids):
    return [list(map(int,POINTS[i])) for i in ids]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--shard',type=int,default=0); ap.add_argument('--shards',type=int,default=1); ap.add_argument('--out',required=True); a=ap.parse_args()
    std=[]
    for j in range(D):
        e=[0]*D;e[j]=1;std.append(INDEX[tuple(e)])
    library=[basis_mask(std)]
    alarms=[]; scanned=0; best=(10**9,None,None); t0=time.time()
    for block in range(a.shard,101,a.shards):
        with urllib.request.urlopen(RAW.format(block),timeout=60) as f:
            text=f.read().decode().strip().splitlines()
        base=int(text[0]); start=block*7498+1
        for off,line in enumerate(text[1:]):
            pid=start+off; rays=decode(line,base); m=ewald_mask(rays); scanned+=1; ec=m.bit_count()
            if ec<best[0]:best=(ec,pid,rays)
            bm=find_basis(m,library,random.Random(pid))
            if bm is None:
                ids=indices(m)
                alarms.append({'id':pid,'ewald_count':ec,'rays':[list(r) for r in rays],
                    'ranks':{str(p):rank_mod(ids,p) for p in (2,3,5,7,11)},'ewald_points':ewald_points(ids)})
            elif bm not in library and len(library)<1024:
                library.append(bm)
        print(json.dumps({'block':block,'scanned':scanned,'bestE':best[0],'bestID':best[1],'alarms':len(alarms),'ray_cache':len(RAY_MASK),'basis_library':len(library),'sec':round(time.time()-t0,2)}),flush=True)
    result={'shard':a.shard,'shards':a.shards,'scanned':scanned,'elapsed_sec':time.time()-t0,
        'min_ewald':best[0],'min_id':best[1],'min_rays':[list(r) for r in best[2]],
        'alarms':alarms,'ray_cache_size':len(RAY_MASK),'basis_library_size':len(library)}
    with open(a.out,'w') as f:json.dump(result,f,separators=(',',':'))
    print('FINAL '+json.dumps({k:v for k,v in result.items() if k not in ('alarms','min_rays')} | {'alarm_count':len(alarms)}),flush=True)

if __name__=='__main__':main()
