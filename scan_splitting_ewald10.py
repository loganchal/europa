import argparse,itertools,json,random,time,math
import numpy as np
D=10
PTS=np.array(list(itertools.product((-1,0,1),repeat=D)),dtype=np.int16)
FULL=(1<<len(PTS))-1
RMASK={}

def nondec(n,lo,hi,maxsum=None,prefix=()):
    if n==0:
        yield prefix;return
    start=prefix[-1] if prefix else lo
    for x in range(start,hi+1):
        if maxsum is not None and sum(prefix)+x>maxsum:break
        yield from nondec(n-1,lo,hi,maxsum,prefix+(x,))

def fano_klein_inputs(d):
    for m in range(1,d):
        for a in nondec(m,0,d-1,maxsum=d-m):yield a

def all_lists(n,lo,hi):
    return itertools.product(range(lo,hi+1),repeat=n)

def dict_order(b,c):
    return all(not (b[i]==b[i+1] and c[i]>c[i+1]) for i in range(len(b)-1))

def bounded_inputs():
    for dp in range(2,10):
        q=10-dp
        for a in fano_klein_inputs(dp):
            m=len(a);r=dp-m;A=r+1-sum(a);B=m+1
            for b in nondec(q,0,B,maxsum=B-1):
                for c in all_lists(q,-A,A):
                    if sum(c)-(q-1)*min(c)>=A:continue
                    if not dict_order(b,c):continue
                    yield dp,a,b,tuple(c)

def fan(dp,a,b,c):
    m=len(a);r=dp-m;q=10-dp
    rays=[];X=[];Y=[];Z=[]
    # X special, then m standard X rays
    X.append(len(rays));r0=np.zeros(D,dtype=int);r0[:m]=-1;r0[dp:]=np.asarray(b,dtype=int);rays.append(r0)
    for i in range(m):
        X.append(len(rays));v=np.zeros(D,dtype=int);v[i]=1;rays.append(v)
    # r standard Y rays, then Y special
    for j in range(r):
        Y.append(len(rays));v=np.zeros(D,dtype=int);v[m+j]=1;rays.append(v)
    Y.append(len(rays));v=np.zeros(D,dtype=int);v[:m]=np.asarray(a,dtype=int);v[m:dp]=-1;v[dp:]=np.asarray(c,dtype=int);rays.append(v)
    # fiber e0 then standard e_i
    Z.append(len(rays));v=np.zeros(D,dtype=int);v[dp:]=-1;rays.append(v)
    for j in range(q):
        Z.append(len(rays));v=np.zeros(D,dtype=int);v[dp+j]=1;rays.append(v)
    R=np.asarray(rays,dtype=int)
    cones=[]
    for ox in X:
      for oy in Y:
       for oz in Z:
        I=tuple(i for i in range(len(R)) if i not in (ox,oy,oz));
        if len(I)!=D:raise RuntimeError('cone size')
        cones.append(I)
    return R,cones

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

def exact_fano(R,cones):
    for I in cones:
        M=R[list(I)]
        if abs(bareiss(M))!=1:return False
        try:x=np.rint(np.linalg.solve(M.astype(float),np.ones(D))).astype(np.int64)
        except Exception:return False
        if not np.array_equal(M@x,np.ones(D,dtype=np.int64)):return False
        inc=set(I);dots=R@x
        for j,z in enumerate(dots):
            if j in inc:
                if z!=1:return False
            elif z>=1:return False
    return True

def rmask(r):
    k=tuple(map(int,r));m=RMASK.get(k)
    if m is None:
        ok=np.abs(PTS@np.asarray(k,dtype=np.int16))<=1
        m=int.from_bytes(np.packbits(ok,bitorder='little').tobytes(),'little')&FULL;RMASK[k]=m
    return m

def emask(R):
    m=FULL
    for r in R:m&=rmask(r)
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
        A[[row,q]]=A[[q,row]];A[row]=(A[row]*pow(int(A[row,col]),-1,p))%p
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

def find_basis(E,seed,trials=3500):
    R=reps(E)
    if len(R)<D:return None
    R=sorted(R,key=lambda i:(int(np.abs(E[i]).sum()),i))
    pools=[R[:min(80,len(R))],R[:min(320,len(R))],R]
    rng=random.Random(seed);per=max(1,trials//len(pools))
    # deterministic combinations from very low-support points first
    p0=pools[0][:min(18,len(pools[0]))]
    for sel in itertools.islice(itertools.combinations(p0,D),3000):
        M=E[list(sel)];fd=round(float(np.linalg.det(M.astype(float))))
        if abs(fd)==1 and abs(bareiss(M))==1:return [int(i) for i in sel]
    for pool in pools:
        if len(pool)<D:continue
        for _ in range(per):
            sel=rng.sample(pool,D);M=E[sel];fd=round(float(np.linalg.det(M.astype(float))))
            if abs(fd)!=1:continue
            if abs(bareiss(M))==1:return [int(i) for i in sel]
    return None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--shards',type=int,required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    t=time.time();bounded=selected=fano=positive=errors=0;rankbad=[];alarms=[];minE=(10**9,None)
    for ordinal,param in enumerate(bounded_inputs()):
        bounded+=1
        if ordinal%a.shards!=a.shard:continue
        selected+=1;dp,aa,b,c=param;R,C=fan(dp,aa,b,c)
        if not exact_fano(R,C):continue
        fano+=1;m=emask(R);I=mask_ids(m);E=PTS[I];ec=len(E)
        if ec<minE[0]:minE=(ec,{'ordinal':ordinal,'dp':dp,'a':aa,'b':b,'c':c,'normals':R.tolist()})
        ranks=[rank_mod(E,p) for p in (2,3,5,7)]
        if min(ranks)<D:
            rec={'ordinal':ordinal,'dp':dp,'a':aa,'b':b,'c':c,'E':ec,'ranks':ranks,'normals':R.tolist()};rankbad.append(rec);print('RANKBAD',json.dumps(rec,separators=(',',':')),flush=True);continue
        bas=find_basis(E,ordinal*1000003+17,5000)
        if bas is None:
            rec={'ordinal':ordinal,'dp':dp,'a':aa,'b':b,'c':c,'E':ec,'ranks':ranks,'normals':R.tolist()};alarms.append(rec);print('ALARM',json.dumps(rec,separators=(',',':')),flush=True)
        else:
            M=E[bas];det=bareiss(M)
            if abs(det)!=1:raise RuntimeError('bad cert')
            positive+=1
        if selected%200==0:print('PROG',json.dumps({'shard':a.shard,'selected':selected,'fano':fano,'positive':positive,'rankbad':len(rankbad),'alarms':len(alarms),'minE':minE[0],'cache':len(RMASK),'sec':round(time.time()-t,1)}),flush=True)
    out={'shard':a.shard,'shards':a.shards,'bounded_total':bounded,'selected':selected,'fano':fano,'positive':positive,'errors':errors,'rankbad':rankbad,'alarms':alarms,'minE':minE,'raymask_cache':len(RMASK),'elapsed':time.time()-t}
    open(a.out,'w').write(json.dumps(out,separators=(',',':')))
    print('FINAL',json.dumps({k:v for k,v in out.items() if k not in ('rankbad','alarms')}|{'rankbad_count':len(rankbad),'alarm_count':len(alarms)},separators=(',',':')),flush=True)
if __name__=='__main__':main()
