import argparse,itertools,json,urllib.request
import numpy as np

D=10
RAW='https://raw.githubusercontent.com/GorrieXIV/Magma/master/libs/data/polytopes/smoothfano/block{}'
GR={k:np.array(list(itertools.product((-1,0,1),repeat=k)),dtype=np.int16) for k in range(2,11)}

def digits(n,b):
 out=[]
 while n:out.append(n%b);n//=b
 return out

def decode(line,base):
 a=digits(int(line),base)
 if len(a)<2:return None
 d=a[0];shift=a[1];c=[z-shift for z in a[2:]]
 if d<2 or d>6 or len(c)%d:return None
 return d,[tuple(c[i:i+d]) for i in range(0,len(c),d)]

def entries():
 ordinal=0
 for block in range(35):
  path=f'/tmp/sf{block}'
  urllib.request.urlretrieve(RAW.format(block),path)
  lines=open(path).read().splitlines();base=int(lines[0])
  for line in lines[1:]:
   if not line.strip():continue
   ordinal+=1
   z=decode(line,base)
   if z:yield ordinal,*z

def rank_mod_stream(base_rays,u,t,k,p=3,want_points=False):
 n=D-k;XB=GR[k];Y=GR[n]
 A=np.asarray(base_rays,dtype=np.int16);ok=np.max(np.abs(XB@A.T),axis=1)<=1;XB=XB[ok]
 sy=Y.sum(axis=1);uu=np.asarray(u,dtype=np.int16)
 basis=[];points=[]
 def add(v):
  nonlocal basis
  a=np.array(v,dtype=np.int64)%p
  for piv,b in basis:
   if a[piv]:a=(a-int(a[piv])*b)%p
  nz=np.flatnonzero(a)
  if len(nz):
   piv=int(nz[0]);a=(a*pow(int(a[piv]),-1,p))%p
   basis.append((piv,a));basis.sort(key=lambda z:z[0])
  return len(basis)
 total=0
 for x in XB:
  s=t*int(np.dot(uu,x));ids=np.flatnonzero(np.abs(s+sy)<=1)
  for j in ids:
   v=np.concatenate([x,Y[j]]);total+=1
   if want_points:points.append(v.copy())
   if add(v)==D and not want_points:return D,total,None
 return len(basis),total,(np.asarray(points,dtype=np.int16) if want_points else None)

def defect3(E):
 # E is centrally symmetric; exact hyperplane defect from rounded ternary DFT, followed by exact check of maximizer.
 n=len(E);shape=(3,)*D;arr=np.zeros(3**D,dtype=np.float64);coords=(E%3).astype(np.intp)
 flat=np.ravel_multi_index(coords.T,shape);arr[flat]=1.0
 F=np.fft.fftn(arr.reshape(shape)).real.ravel();F[0]=-1e100;j=int(np.argmax(F));mx=int(round(float(F[j])))
 defect=2*(n-mx)//3;a=np.array(np.unravel_index(j,shape),dtype=np.int64)
 exact=int(np.count_nonzero((E@a)%3))
 if exact!=defect:defect=exact
 return int(defect),a.tolist()

def candidate_rays(base,u,t,k):
 n=D-k;R=[tuple(r)+(0,)*n for r in base]
 for j in range(n):R.append((0,)*k+(0,)*j+(-1,)+(0,)*(n-j-1))
 R.append(tuple(int(t*q) for q in u)+(1,)*n)
 return R

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--shards',type=int,required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 tested=0;boundary_collapse=[];next_collapse=0;best=(10**9,None);bydim={}
 for oid,k,base in entries():
  if (oid-1)%a.shards!=a.shard:continue
  bydim[k]=bydim.get(k,0)+1;n=D-k
  for ui,u in enumerate(base):
   tested+=1
   r,total,_=rank_mod_stream(base,u,n,k,3,False)
   rn,tn,_=rank_mod_stream(base,u,n+1,k,3,False)
   if rn<D:next_collapse+=1
   if r<D:
    rr=rank_mod_stream(base,u,n,k,3,True);E=rr[2];d3,avec=defect3(E)
    rec={'oid':oid,'base_dim':k,'base_rays':base,'u_idx':ui,'u':u,'t':n,'rank3':r,'E':len(E),'defect3':d3,'a3':avec,'rays':candidate_rays(base,u,n,k)}
    boundary_collapse.append(rec);print('BOUNDARY COLLAPSE',json.dumps(rec,separators=(',',':')),flush=True)
   elif rn<D:
    # exact defect on the last Fano-side twist n, since this is the closest arithmetic threshold.
    rr=rank_mod_stream(base,u,n,k,3,True);E=rr[2];d3,avec=defect3(E)
    if d3<best[0]:
     best=(d3,{'oid':oid,'base_dim':k,'base_rays':base,'u_idx':ui,'u':u,'t':n,'rank3':r,'E':len(E),'defect3':d3,'a3':avec,'rays':candidate_rays(base,u,n,k)})
     print('BEST THRESHOLD',json.dumps(best[1],separators=(',',':')),flush=True)
  if sum(bydim.values())%100==0:print('PROG',a.shard,sum(bydim.values()),tested,'nextcollapse',next_collapse,'best',best[0],flush=True)
 out={'shard':a.shard,'shards':a.shards,'bases_by_dim':bydim,'tested':tested,'boundary_collapse':boundary_collapse,'next_collapse':next_collapse,'best':best[1]}
 open(a.out,'w').write(json.dumps(out,separators=(',',':')))
 print('FINAL',json.dumps({'shard':a.shard,'bases':sum(bydim.values()),'tested':tested,'boundary':len(boundary_collapse),'nextcollapse':next_collapse,'best':best[0]},separators=(',',':')),flush=True)
if __name__=='__main__':main()
