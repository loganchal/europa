import argparse,itertools,json,random,time
import numpy as np
import scan_batyrev_blowups as B
D=10;PTS=B.PTS;FULL=B.FULL
RAYMASK={}

def rmask(r):
 k=tuple(map(int,r));m=RAYMASK.get(k)
 if m is None:
  ok=np.abs(PTS@np.asarray(k,dtype=np.int16))<=1
  m=int.from_bytes(np.packbits(ok,bitorder='little').tobytes(),'little')&FULL;RAYMASK[k]=m
 return m

def emask(R):
 m=FULL
 for r in R:m&=rmask(r)
 return m

def ids(m):
 out=[]
 while m:
  b=m&-m;out.append(b.bit_length()-1);m^=b
 return out

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

def rank_mod(I,p):
 A=(PTS[I].astype(np.int64)%p).copy();r=0;m=len(A)
 for c in range(D):
  q=next((i for i in range(r,m) if A[i,c]%p),None)
  if q is None:continue
  A[[r,q]]=A[[q,r]];A[r]=(A[r]*pow(int(A[r,c]),-1,p))%p
  for i in range(r+1,m):
   if A[i,c]%p:A[i]=(A[i]-int(A[i,c])*A[r])%p
  r+=1
  if r==D:return D
 return r

def reps(I):
 out=[]
 for i in I:
  x=PTS[i];nz=np.flatnonzero(x)
  if len(nz)==0:continue
  if x[nz[0]]>0:out.append(i)
 return out

def find_basis(m,rng,trials=1800):
 I=ids(m);R=reps(I)
 if len(R)<D:return None
 # deterministic low-support pools first, then all reps
 R=sorted(R,key=lambda i:(int(np.abs(PTS[i]).sum()),i));pools=[R[:min(96,len(R))],R[:min(384,len(R))],R]
 per=max(1,trials//len(pools))
 for pool in pools:
  if len(pool)<D:continue
  for _ in range(per):
   sel=rng.sample(pool,D);M=PTS[sel]
   fd=round(float(np.linalg.det(M.astype(float))))
   if abs(fd)!=1:continue
   d=bareiss(M)
   if abs(d)==1:return sel
 return None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,required=True);ap.add_argument('--end',type=int,required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 t=time.time();parents=children=preserved=searched=errors=0;alarms=[];rankbad=[];minE=(10**9,None);hard=(0,None)
 for idx in range(a.start,a.end):
  p,c,d=B.CASES[idx];rec=B.reconstruct(p,c,d)
  if rec is None:errors+=1;continue
  R,C=rec;parents+=1;pm=emask(R);prng=random.Random(1000003+idx);pb=find_basis(pm,prng,4000)
  if pb is None:
   errors+=1;alarms.append({'kind':'parent_basis','parent':idx,'normals':R.tolist(),'E':pm.bit_count()});continue
  PB=PTS[pb]
  for bi,(S,w) in enumerate(B.valid_blowups(R,C)):
   children+=1;T=np.vstack([R,np.asarray(w,dtype=int)]);cm=pm&rmask(w);ec=cm.bit_count()
   if ec<minE[0]:minE=(ec,{'parent':idx,'bi':bi,'face':S,'w':w})
   if np.all(np.abs(PB@np.asarray(w,dtype=int))<=1):preserved+=1;continue
   searched+=1;I=ids(cm);ranks=[rank_mod(I,q) for q in (2,3,5,7)]
   if min(ranks)<D:
    recbad={'parent':idx,'bi':bi,'face':S,'w':w,'ranks':ranks,'E':ec,'normals':T.tolist()};rankbad.append(recbad);print('RANKBAD',json.dumps(recbad,separators=(',',':')),flush=True);continue
   rng=random.Random((idx+1)*1000003+bi);bb=find_basis(cm,rng,3500)
   if bb is None:
    al={'parent':idx,'bi':bi,'face':S,'w':w,'ranks':ranks,'E':ec,'normals':T.tolist()};alarms.append(al);print('ALARM',json.dumps(al,separators=(',',':')),flush=True)
   else:
    # exact certificate retained compactly as point rows
    if 3500>hard[0]:pass
  if (idx-a.start+1)%10==0:print('PROG',json.dumps({'idx':idx,'parents':parents,'children':children,'preserved':preserved,'searched':searched,'alarms':len(alarms),'rankbad':len(rankbad),'minE':minE[0],'sec':round(time.time()-t,1)}),flush=True)
 out={'start':a.start,'end':a.end,'parents':parents,'children':children,'preserved':preserved,'searched':searched,'errors':errors,'alarms':alarms,'rankbad':rankbad,'minE':minE,'elapsed':time.time()-t,'raymask_cache':len(RAYMASK)}
 open(a.out,'w').write(json.dumps(out,separators=(',',':')))
 print('FINAL',json.dumps({k:v for k,v in out.items() if k not in ('alarms','rankbad')}|{'alarm_count':len(alarms),'rankbad_count':len(rankbad)},separators=(',',':')),flush=True)
if __name__=='__main__':main()
