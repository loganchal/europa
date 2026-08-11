import itertools,json,heapq,time,argparse
import numpy as np
from scipy.spatial import ConvexHull
D=10
PTS=np.array(list(itertools.product((-1,0,1),repeat=D)),dtype=np.int16);FULL=(1<<len(PTS))-1
MOD=PTS%2
HM=[]
def bm(ok):return int.from_bytes(np.packbits(ok,bitorder='little').tobytes(),'little')&FULL
for code in range(1,1<<D):
 v=np.array([(code>>i)&1 for i in range(D)],dtype=np.int16);HM.append((code,bm((MOD@v)%2==1)))
RC={}
def rmask(r):
 k=tuple(map(int,r));q=RC.get(k)
 if q is None:q=bm(np.abs(PTS@np.array(k,dtype=np.int16))<=1);RC[k]=q
 return q
def emask(R):
 m=FULL
 for r in R:m&=rmask(r)
 return m
def score(m):
 q,c=min(((m&h).bit_count(),code) for code,h in HM);return q,m.bit_count(),c

def bareiss(M):
 A=[list(map(int,r)) for r in M];n=len(A);prev=1;sgn=1
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
def invu(M):
 if abs(bareiss(M))!=1:return None
 X=np.rint(np.linalg.inv(np.array(M,dtype=float))).astype(int)
 return X if np.array_equal(np.array(M,dtype=int)@X,np.eye(D,dtype=int)) else None

def smooth(R):
 if len(set(map(tuple,R)))!=len(R):return None
 try:h=ConvexHull(R.astype(float),qhull_options='Qt')
 except:return None
 if len(h.vertices)!=len(R):return None
 seen=set();fac=[]
 for I0 in h.simplices:
  I=tuple(sorted(map(int,I0)));M=R[list(I)];X=invu(M)
  if X is None:return None
  u=X@np.ones(D,dtype=int);vals=R@u
  eq=tuple(np.flatnonzero(vals==1).tolist())
  if len(eq)!=D:return None
  if np.any(vals[[j for j in range(len(R)) if j not in eq]]>=1):return None
  if eq not in seen:seen.add(eq);fac.append(eq)
 return fac

def standard(R):
 C=smooth(R)
 if C is None:return None
 F=C[0];X=invu(R[list(F)]);T=R@X
 order=list(F)+[i for i in range(len(R)) if i not in set(F)];T=T[order]
 G=smooth(T)
 return (T,G) if G is not None else None

def key(R):return tuple(sorted(tuple(map(int,r)) for r in R))
def faces(C):
 out=set()
 for c in C:
  for k in range(2,D+1):out.update(itertools.combinations(c,k))
 return out

def prep(R,C):
 nr=len(R);nc=len(C);H=np.zeros((nc,nr),dtype=int);A=[];pos=[];CM=[]
 for ci,c in enumerate(C):
  X=invu(R[list(c)]);H[ci]=R@(X@np.ones(D,dtype=int));A.append(R@X);pos.append({r:p for p,r in enumerate(c)})
  z=0
  for r in c:z|=1<<r
  CM.append(z)
 return H,A,pos,CM

def stars(R,C):
 nr=len(R);nc=len(C);H,A,pos,CM=prep(R,C);rset=set(map(tuple,R));out=[]
 for S in faces(C):
  w=R[list(S)].sum(axis=0)
  if tuple(w) in rset:continue
  k=len(S);sm=sum(1<<s for s in S);aff=[ci for ci,z in enumerate(CM) if z&sm==sm];aset=set(aff)
  hs=H[:,list(S)].sum(axis=1)
  if any(hs[ci]>=1 for ci in range(nc) if ci not in aset):continue
  ok=True
  for ci in aff:
   outside=[j for j in range(nr) if not (CM[ci]>>j)&1]
   for s in S:
    if np.any(H[ci,outside]+(1-k)*A[ci][outside,pos[ci][s]]>=1):ok=False;break
   if not ok:break
  if not ok:continue
  wi=nr;SS=set(S);G=[]
  for c in C:
   cs=set(c)
   if SS.issubset(cs):
    for s in SS:G.append(tuple(sorted((cs-{s})|{wi})))
   else:G.append(tuple(c))
  out.append((w,list(dict.fromkeys(G)),S))
 return out

ROOT=np.array([[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1],[0,0,0,0,2,0,-1,0,-1,0],[-1,-1,-1,-1,5,0,0,0,0,-1],[0,0,0,0,-1,-1,0,-1,0,0],[0,0,0,0,-1,0,0,0,0,0],[0,0,0,0,-1,1,0,0,0,0]],dtype=int)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--raycap',type=int,default=17);ap.add_argument('--seconds',type=int,default=2400);ap.add_argument('--out',required=True);a=ap.parse_args()
 g=standard(ROOT);assert g;R,C=g;sc=score(emask(R));heap=[];seen={key(R)};cnt=0
 def push(T,G,h):
  nonlocal cnt
  s=score(emask(T));cnt+=1;heapq.heappush(heap,(s[0],s[1],len(T),cnt,T,G,h))
 push(R,C,[]);best=(sc,R,C,[]);processed=0;t=time.time()
 while heap and time.time()-t<a.seconds and len(seen)<50000:
  p,e,n,_,R,C,hist=heapq.heappop(heap);processed+=1
  if (p,e)<best[0][:2]:
   best=((p,e,score(emask(R))[2]),R,C,hist);print('BEST',best[0],'rays',n,'processed',processed,'seen',len(seen),flush=True)
   if p==0:break
  for di in range(n):
   g=standard(np.delete(R,di,axis=0))
   if not g:continue
   T,G=g;k=key(T)
   if k in seen:continue
   seen.add(k);push(T,G,hist+[{'op':'del','i':di}])
  if n<a.raycap:
   for w,G,S in stars(R,C):
    T=np.vstack([R,w]);k=key(T)
    if k in seen:continue
    seen.add(k);push(T,G,hist+[{'op':'add','face':list(S),'w':list(map(int,w))}])
  if processed%100==0:print('PROG',processed,len(seen),len(heap),'top',heap[0][:3] if heap else None,'sec',round(time.time()-t,1),flush=True)
 out={'score':best[0],'rays':best[1].tolist(),'cones':[list(c) for c in best[2]],'hist':best[3],'processed':processed,'seen':len(seen),'queue':len(heap),'elapsed':time.time()-t,'raycap':a.raycap}
 with open(a.out,'w') as f:json.dump(out,f,separators=(',',':'))
 print('FINAL',json.dumps({k:v for k,v in out.items() if k not in ('rays','cones','hist')}),flush=True)
if __name__=='__main__':main()
