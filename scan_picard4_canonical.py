import argparse,ast,glob,itertools,json,os,random,re
import numpy as np
from scipy.spatial import ConvexHull

D=10
POINTS=np.array(list(itertools.product((-1,0,1),repeat=D)),dtype=np.int16)

def det_bareiss(A):
    A=[list(map(int,r)) for r in A]; n=len(A)
    if n==0:return 1
    s=1; prev=1
    for k in range(n-1):
        if A[k][k]==0:
            for i in range(k+1,n):
                if A[i][k]: A[k],A[i]=A[i],A[k]; s=-s; break
            else:return 0
        piv=A[k][k]
        for i in range(k+1,n):
            for j in range(k+1,n):
                A[i][j]=(A[i][j]*piv-A[i][k]*A[k][j])//prev
        prev=piv
        for i in range(k+1,n):A[i][k]=0
    return s*A[-1][-1]

def parse_matrix(txt):
    s=''.join(txt) if isinstance(txt,list) else str(txt)
    a=s.find('Matrix(')
    if a<0:return None
    s=s[a+7:]
    depth=1; end=None
    for i,ch in enumerate(s):
        if ch=='(':depth+=1
        elif ch==')':
            depth-=1
            if depth==0:end=i;break
    if end is None:return None
    try:
        M=ast.literal_eval(s[:end])
        if not M or not isinstance(M[0],list):return None
        return np.array(M,dtype=int)
    except:return None

def seed_maps(root):
    out=[]
    for f in glob.glob(os.path.join(root,'fan-giving maps','*.ipynb')):
        try:nb=json.load(open(f))
        except:continue
        src='\n'.join(''.join(c.get('source',[])) for c in nb.get('cells',[]) if c.get('cell_type')=='code')
        mm=re.search(r'fanlikes_index\s*=\s*(\d+)',src)
        if not mm:continue
        idx=int(mm.group(1))
        for c in nb.get('cells',[]):
            if c.get('cell_type')!='code' or 'fans.append(C.char)' not in ''.join(c.get('source',[])):continue
            M=None
            for o in c.get('outputs',[]):
                dat=o.get('data',{})
                if 'text/plain' in dat:
                    M=parse_matrix(dat['text/plain'])
                    if M is not None:break
            if M is not None and M.ndim==2 and M.shape[1]-M.shape[0]==4:
                out.append((idx,M))
    uniq={}
    for idx,M in out:uniq[(idx,tuple(map(tuple,M.tolist())))]=M
    return [(k[0],M) for k,M in uniq.items()]

def comps(total,n,prefix=()):
    if n==1:
        yield prefix+(total,);return
    for x in range(total+1):yield from comps(total-x,n-1,prefix+(x,))

def wedge(U,idx):
    U=np.asarray(U,dtype=int); m,n=U.shape
    V=np.zeros((m+1,n+1),dtype=int)
    V[:m,:n]=U
    V[idx,n]=1
    V[m,n]=-1
    return V

def multiwedge(M,counts):
    U=M.T.copy()
    for i,c in enumerate(counts):
        for _ in range(c):U=wedge(U,i)
    return U

def fano_basis(U):
    try:h=ConvexHull(U.astype(float),qhull_options='Qt')
    except:return None
    if set(map(int,h.vertices))!=set(range(len(U))):return None
    facets=np.unique(np.sort(h.simplices,axis=1),axis=0)
    first=None
    for I in facets:
        M=U[I]; d=det_bareiss(M)
        if abs(d)!=1:return None
        inv=np.rint(np.linalg.inv(M)).astype(int)
        if not np.array_equal(M@inv,np.eye(D,dtype=int)):return None
        ell=inv@np.ones(D,dtype=int)
        vals=U@ell
        if np.any(vals>1):return None
        eq=np.flatnonzero(vals==1)
        if len(eq)!=D or set(map(int,eq))!=set(map(int,I)):return None
        if first is None:first=(M,inv)
    return first

def rank_mod(A,p):
    A=np.asarray(A,dtype=np.int64)%p
    if len(A)==0:return 0
    A=A.copy();r=0
    for c in range(A.shape[1]):
        piv=next((i for i in range(r,len(A)) if A[i,c]%p),None)
        if piv is None:continue
        A[[r,piv]]=A[[piv,r]]
        inv=pow(int(A[r,c]),-1,p);A[r]=(A[r]*inv)%p
        for i in range(len(A)):
            if i!=r and A[i,c]%p:A[i]=(A[i]-A[i,c]*A[r])%p
        r+=1
        if r==A.shape[1]:break
    return r

def ewald(U,basis):
    M,inv=basis
    V=U@inv
    if not np.array_equal(V[:0],V[:0]):raise RuntimeError
    vals=POINTS@V.T
    E=POINTS[np.all(np.abs(vals)<=1,axis=1)]
    return V,E

def find_basis(E,rng,trials=1500):
    if len(E)<D:return None
    ids=list(range(len(E)))
    for _ in range(trials):
        I=rng.sample(ids,D); d=det_bareiss(E[I])
        if abs(d)==1:return [E[i].tolist() for i in I],d
    return None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);ap.add_argument('--out',required=True);a=ap.parse_args()
    maps=seed_maps(a.root); candidates=fano=positive=0;rankbad=[];alarms=[];minE=(10**9,None); global_i=0
    for si,M in maps:
        n,m=M.shape;k=D-n
        if k<0:continue
        for cnt in comps(k,m):
            gi=global_i;global_i+=1
            if gi%a.shards!=a.shard:continue
            candidates+=1
            U=multiwedge(M,cnt)
            if U.shape!=(14,10):continue
            fb=fano_basis(U)
            if fb is None:continue
            fano+=1;V,E=ewald(U,fb)
            if len(E)<minE[0]:minE=(len(E),{'seed':si,'counts':cnt,'normals':V.tolist()});print('BEST',minE[0],si,cnt,flush=True)
            ranks={p:rank_mod(E,p) for p in (2,3,5,7,11)}
            rec={'seed':si,'counts':cnt,'E':len(E),'ranks':ranks,'normals':V.tolist()}
            if min(ranks.values())<D:
                rankbad.append(rec);print('RANKBAD '+json.dumps(rec,separators=(',',':')),flush=True);break
            got=find_basis(E,random.Random(1000003+gi))
            if got is None:
                alarms.append(rec);print('ALARM '+json.dumps(rec,separators=(',',':')),flush=True)
            else:positive+=1
        if rankbad:break
    out={'shard':a.shard,'shards':a.shards,'seed_maps':len(maps),'global_candidates':global_i,'candidates':candidates,'fano':fano,'positive':positive,'rankbad':rankbad,'alarms':alarms,'minE':minE}
    json.dump(out,open(a.out,'w'),separators=(',',':'))
    print('FINAL '+json.dumps({'shard':a.shard,'maps':len(maps),'global':global_i,'candidates':candidates,'fano':fano,'positive':positive,'rankbad':len(rankbad),'alarms':len(alarms),'minE':minE[0]},separators=(',',':')),flush=True)
if __name__=='__main__':main()
