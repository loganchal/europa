import argparse,itertools,json,random,time
import numpy as np
D=10; N=14
PTS=np.array(list(itertools.product((-1,0,1),repeat=D)),dtype=np.int16)
FULL=(1<<len(PTS))-1
RMASK={}

def blocks(path):
    cur=[]
    with open(path) as f:
        for line in f:
            s=line.strip()
            if not s:
                if cur:
                    if len(cur)!=N: raise ValueError(('block',len(cur)))
                    yield np.array(cur,dtype=int);cur=[]
                continue
            v=[int(x) for x in s.split()]
            if len(v)!=D: raise ValueError(('row',s))
            cur.append(v)
    if cur:
        if len(cur)!=N: raise ValueError(('block',len(cur)))
        yield np.array(cur,dtype=int)

def bareiss(M):
    A=[list(map(int,row)) for row in M];n=len(A);prev=1;sgn=1
    for k in range(n-1):
        if A[k][k]==0:
            q=next((i for i in range(k+1,n) if A[i][k]),None)
            if q is None:return 0
            A[k],A[q]=A[q],A[k];sgn=-sgn
        piv=A[k][k]
        for i in range(k+1,n):
            aik=A[i][k]
            for j in range(k+1,n):A[i][j]=(A[i][j]*piv-aik*A[k][j])//prev
        prev=piv
        for i in range(k+1,n):A[i][k]=0
    return sgn*A[-1][-1]

def rmask(r):
    k=tuple(map(int,r));m=RMASK.get(k)
    if m is None:
        vals=PTS@np.asarray(k,dtype=np.int16)
        ok=np.abs(vals)<=1
        m=int.from_bytes(np.packbits(ok,bitorder='little').tobytes(),'little')&FULL
        RMASK[k]=m
    return m

def emask(R):
    m=FULL
    for r in R:
        m &= rmask(r)
        if not m: break
    return m

def mask_ids(m):
    out=[]
    while m:
        b=m&-m;out.append(b.bit_length()-1);m^=b
    return out

def rank_mod(E,p):
    A=(E.astype(np.int64)%p).copy();nr=len(A);row=0
    for col in range(D):
        q=next((i for i in range(row,nr) if A[i,col]%p),None)
        if q is None:continue
        A[[row,q]]=A[[q,row]]
        A[row]=(A[row]*pow(int(A[row,col]),-1,p))%p
        for i in range(row+1,nr):
            if A[i,col]%p:A[i]=(A[i]-int(A[i,col])*A[row])%p
        row+=1
        if row==D:return D
    return row

def reps(E):
    out=[]
    for i,x in enumerate(E):
        nz=np.flatnonzero(x)
        if len(nz) and x[nz[0]]>0:out.append(i)
    return out

def find_basis(E,seed,trials=5000):
    R=reps(E)
    if len(R)<D:return None
    R=sorted(R,key=lambda i:(int(np.abs(E[i]).sum()),i))
    low=R[:min(20,len(R))]
    for sel in itertools.islice(itertools.combinations(low,D),6000):
        M=E[list(sel)]
        fd=round(float(np.linalg.det(M.astype(float))))
        if abs(fd)==1 and abs(bareiss(M))==1:return list(map(int,sel))
    rng=random.Random(seed)
    pools=[R[:min(100,len(R))],R[:min(400,len(R))],R]
    per=max(1,trials//len(pools))
    for pool in pools:
        if len(pool)<D:continue
        for _ in range(per):
            sel=rng.sample(pool,D);M=E[sel]
            fd=round(float(np.linalg.det(M.astype(float))))
            if abs(fd)==1 and abs(bareiss(M))==1:return list(map(int,sel))
    return None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--shard',type=int,required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    t=time.time();total=positive=errors=0;rankbad=[];alarms=[];minE=(10**9,None)
    for idx,R in enumerate(blocks(a.input),1):
        total+=1
        # Obro special embedding starts with the standard basis. Verify this before using the 3^10 cube.
        if not np.array_equal(R[:D],np.eye(D,dtype=int)):
            errors+=1
            print('ERROR nonstandard',idx,R[:D].tolist(),flush=True);continue
        m=emask(R);E=PTS[mask_ids(m)];ec=len(E)
        if ec<minE[0]:minE=(ec,{'index':idx,'normals':R.tolist()})
        ranks=[rank_mod(E,p) for p in (2,3,5,7,11)]
        if min(ranks)<D:
            rec={'index':idx,'E':ec,'ranks':ranks,'normals':R.tolist()};rankbad.append(rec)
            print('RANKBAD '+json.dumps(rec,separators=(',',':')),flush=True);continue
        bas=find_basis(E,a.shard*1000003+idx*9176+17)
        if bas is None:
            rec={'index':idx,'E':ec,'ranks':ranks,'normals':R.tolist()};alarms.append(rec)
            print('ALARM '+json.dumps(rec,separators=(',',':')),flush=True)
        else:
            det=bareiss(E[bas])
            if abs(det)!=1:raise RuntimeError('bad determinant certificate')
            positive+=1
        if total%250==0:
            print('PROG '+json.dumps({'shard':a.shard,'total':total,'positive':positive,'rankbad':len(rankbad),'alarms':len(alarms),'minE':minE[0],'cache':len(RMASK),'sec':round(time.time()-t,1)},separators=(',',':')),flush=True)
    out={'shard':a.shard,'total':total,'positive':positive,'errors':errors,'rankbad':rankbad,'alarms':alarms,'minE':minE,'mask_cache':len(RMASK),'elapsed':time.time()-t}
    with open(a.out,'w') as f:json.dump(out,f,separators=(',',':'))
    print('FINAL '+json.dumps({'shard':a.shard,'total':total,'positive':positive,'errors':errors,'rankbad_count':len(rankbad),'alarm_count':len(alarms),'minE':minE[0],'elapsed':out['elapsed']},separators=(',',':')),flush=True)

if __name__=='__main__':main()
