import argparse,itertools,json,time
import numpy as np,requests

D=6
CUBE=np.array(list(itertools.product((-1,0,1),repeat=D)),dtype=np.int16)
URL='https://polydb.org/rest/current/find/Polytopes/Lattice/SmoothReflexive'

def inv_unimod(M):
 M=np.asarray(M,dtype=int)
 X=np.rint(np.linalg.inv(M.astype(float))).astype(int)
 if not np.array_equal(M@X,np.eye(D,dtype=int)): raise ValueError('non-unimodular cone')
 return X

def unpack_matrix(x):
 if isinstance(x,dict):
  for k in ('data','value','dense','matrix'):
   if k in x:return unpack_matrix(x[k])
  raise TypeError(('matrix wrapper',x))
 if not isinstance(x,list):raise TypeError(type(x))
 if x and isinstance(x[-1],dict) and set(x[-1])=={'cols'}:
  x=x[:-1]
 return x

def fetch_page(skip):
 p={'query':json.dumps({'DIM':D},separators=(',',':')),'limit':10,'skip':skip,'sort':json.dumps({'_id':1},separators=(',',':'))}
 r=requests.get(URL,params=p,timeout=60);r.raise_for_status();return r.json()

def scan(doc):
 # PolyDB stores the smooth polytope itself. FACETS are homogeneous facet
 # inequalities [1,a], while VERTICES_IN_FACETS is facet -> vertex incidence.
 # The normal fan rays are the facet normals a (global sign is irrelevant to E),
 # and each polytope vertex gives one maximal fan cone via its incident facets.
 V=np.asarray(unpack_matrix(doc['VERTICES']),dtype=int)
 F=np.asarray(unpack_matrix(doc['FACETS']),dtype=int)
 if F.ndim!=2 or F.shape[1]!=D+1:raise ValueError(('facet shape',F.shape))
 R0=F[:,1:].astype(int)
 vif=doc.get('VERTICES_IN_FACETS')
 if vif is None:
  H=V if V.shape[1]==D+1 else np.column_stack([np.ones(len(V),dtype=int),V])
  vals=F@H.T
  vif=[np.flatnonzero(vals[i]==0).tolist() for i in range(len(F))]
 else:vif=unpack_matrix(vif)
 vif=[tuple(map(int,z)) for z in vif]
 nverts=len(V)
 cones=[]
 for j in range(nverts):
  C=tuple(i for i,row in enumerate(vif) if j in row)
  if len(C)!=D:raise ValueError(('not simple',doc['_id'],j,len(C),C))
  cones.append(C)
 cones=sorted(set(cones))
 B=R0[list(cones[0])]
 X=inv_unimod(B)
 R=R0@X
 if not np.array_equal(R[list(cones[0])],np.eye(D,dtype=int)):raise ValueError('basis normalization')
 vals=CUBE@R.T
 ok=np.max(np.abs(vals),axis=1)<=1
 E=CUBE[ok]; EV=vals[ok]
 faces=set()
 for C in cones:
  for k in range(1,D+1):faces.update(itertools.combinations(C,k))
 bad=[]
 for I in faces:
  A=EV[:,I]
  good=np.any((np.sum(A==1,axis=1)==1)&np.all(A>=0,axis=1)&np.all(A<=1,axis=1))
  if not good:bad.append(I)
 return {'id':doc['_id'],'n_rays':len(R),'n_maxcones':len(cones),'E':len(E),'bad_faces':[list(x) for x in bad],
         'normals':R.tolist() if bad else None,'maxcones':[list(x) for x in cones] if bad else None}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,required=True);ap.add_argument('--end',type=int,required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 t=time.time();fails=[];errors=[];seen=[];n=0
 for s in range(a.start,a.end,10):
  try:docs=fetch_page(s)
  except Exception as e:errors.append({'skip':s,'error':repr(e)});continue
  for doc in docs[:max(0,a.end-s)]:
   n+=1;seen.append(doc.get('_id'))
   try:
    q=scan(doc)
    if q['bad_faces']:
     fails.append(q);print('STARFAIL',json.dumps({k:v for k,v in q.items() if k not in ('normals','maxcones')},separators=(',',':')),flush=True)
   except Exception as e:errors.append({'id':doc.get('_id'),'error':repr(e)})
  if (s-a.start)%100==0:print('PROG',s,n,len(fails),len(errors),flush=True)
 out={'start':a.start,'end':a.end,'scanned':n,'first_id':seen[0] if seen else None,'last_id':seen[-1] if seen else None,'fail_count':len(fails),'fails':fails,'errors':errors,'elapsed':time.time()-t}
 open(a.out,'w').write(json.dumps(out,separators=(',',':')))
 print('FINAL',json.dumps({k:v for k,v in out.items() if k!='fails'},separators=(',',':')),flush=True)
 if errors:raise SystemExit(2)
if __name__=='__main__':main()
