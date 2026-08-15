import argparse,gzip,itertools,json,random,tempfile,urllib.request
import numpy as np
D=9; DD=10
PTS=np.array(list(itertools.product((-1,0,1),repeat=DD)),dtype=np.int16)
URL='https://polymake.org/polytopes/paffenholz/data/smooth-fano/d9/fv-09-{:02d}p.gz'

def bareiss(A):
 A=[list(map(int,r)) for r in A];n=len(A);sg=1;prev=1
 for k in range(n-1):
  if A[k][k]==0:
   q=next((i for i in range(k+1,n) if A[i][k]),None)
   if q is None:return 0
   A[k],A[q]=A[q],A[k];sg=-sg
  p=A[k][k]
  for i in range(k+1,n):
   for j in range(k+1,n):A[i][j]=(A[i][j]*p-A[i][k]*A[k][j])//prev
  prev=p
  for i in range(k+1,n):A[i][k]=0
 return sg*A[-1][-1]

def invu(A):
 X=np.rint(np.linalg.inv(np.asarray(A,dtype=float))).astype(int)
 return X if np.array_equal(np.asarray(A,dtype=int)@X,np.eye(len(A),dtype=int)) else None

def records(path):
 cur=[]
 with gzip.open(path,'rt') as f:
  for raw in f:
   s=raw.strip()
   if s=='FACETS':
    if cur:yield np.array(cur,dtype=int);cur=[]
    continue
   if not s:continue
   z=[int(x) for x in s.split()]
   if len(z)==10 and z[0]==1:cur.append([-q for q in z[1:]])
  if cur:yield np.array(cur,dtype=int)

def getbase(facets,index):
 p=tempfile.NamedTemporaryFile(suffix='.gz',delete=False).name;urllib.request.urlretrieve(URL.format(facets),p)
 for i,U in enumerate(records(p),1):
  if i==index:return U
 raise ValueError(index)

def base_facets(U):
 from scipy.spatial import ConvexHull
 h=ConvexHull(U.astype(float),qhull_options='Qt');out=[]
 for I in np.unique(np.sort(h.simplices,axis=1),axis=0):
  if len(I)!=9:continue
  M=U[I];X=invu(M)
  if X is None:raise ValueError('base nonunimodular')
  v=X@np.ones(9,dtype=int);vals=U@v;eq=np.flatnonzero(vals==1)
  if len(eq)==9 and set(eq)==set(I) and np.all(vals[[j for j in range(len(U)) if j not in I]]<1):out.append((I,X))
 return out

def bundle_ok(U,cones,a):
 # a length m; normalized first9 zero. Check both pole facets over every base facet.
 m=len(U)
 for I,X in cones:
  for s in (-1,1):
   rhs=np.ones(9,dtype=int)-s*a[I]
   vx=X@rhs
   vals=U@vx+s*a
   inc=np.zeros(m,dtype=bool);inc[I]=True
   if np.any(vals[~inc]>=1):return False
 return True

def rankmod(A,p):
 A=np.asarray(A,dtype=np.int64)%p;r=0
 for c in range(A.shape[1]):
  q=next((i for i in range(r,len(A)) if A[i,c]%p),None)
  if q is None:continue
  A[[r,q]]=A[[q,r]];iv=pow(int(A[r,c]),-1,p);A[r]=(A[r]*iv)%p
  for i in range(len(A)):
   if i!=r and A[i,c]%p:A[i]=(A[i]-A[i,c]*A[r])%p
  r+=1
  if r==A.shape[1]:break
 return r

def ewald(U,a):
 N=np.zeros((len(U)+2,10),dtype=int);N[:len(U),:9]=U;N[:len(U),9]=a;N[-2,9]=1;N[-1,9]=-1
 vals=PTS@N.T;return PTS[np.all(np.abs(vals)<=1,axis=1)],N

def findbasis(E,rng,trials=1000):
 if len(E)<10:return None
 ids=range(len(E))
 for _ in range(trials):
  I=rng.sample(ids,10);d=bareiss(E[I])
  if abs(d)==1:return d
 return None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--facets',type=int,required=True);ap.add_argument('--index',type=int,required=True);ap.add_argument('--R',type=int,default=12);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);ap.add_argument('--out',required=True);q=ap.parse_args()
 U=getbase(q.facets,q.index);assert np.array_equal(U[:9],np.eye(9,dtype=int));cones=base_facets(U);r=len(U)-9
 tested=fano=positive=0;rankbad=[];alarms=[];boundary=0;minE=(10**9,None);ordn=0
 grid=range(-q.R,q.R+1)
 for ex in itertools.product(grid,repeat=r):
  oi=ordn;ordn+=1
  if oi%q.shards!=q.shard:continue
  tested+=1;a=np.zeros(len(U),dtype=int);a[9:]=ex
  if not bundle_ok(U,cones,a):continue
  fano+=1
  if any(abs(x)==q.R for x in ex):boundary+=1
  E,N=ewald(U,a);ranks={p:rankmod(E,p) for p in (2,3,5,7,11)}
  rec={'twist':ex,'E':len(E),'ranks':ranks,'normals':N.tolist()}
  if len(E)<minE[0]:minE=(len(E),rec);print('BEST',q.facets,q.index,len(E),ex,flush=True)
  if min(ranks.values())<10:
   rankbad.append(rec);print('RANKBAD '+json.dumps(rec,separators=(',',':')),flush=True);break
  if findbasis(E,random.Random(1000003+oi)) is None:
   alarms.append(rec);print('ALARM '+json.dumps(rec,separators=(',',':')),flush=True)
  else:positive+=1
 out={'facets':q.facets,'index':q.index,'R':q.R,'shard':q.shard,'shards':q.shards,'twist_dim':r,'grid_total':ordn,'tested':tested,'fano':fano,'positive':positive,'boundary':boundary,'rankbad':rankbad,'alarms':alarms,'minE':minE}
 json.dump(out,open(q.out,'w'),separators=(',',':'))
 print('FINAL '+json.dumps({'facets':q.facets,'index':q.index,'shard':q.shard,'tested':tested,'fano':fano,'positive':positive,'boundary':boundary,'rankbad':len(rankbad),'alarms':len(alarms),'minE':minE[0]},separators=(',',':')),flush=True)
if __name__=='__main__':main()
