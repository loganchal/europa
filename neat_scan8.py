import argparse,itertools,json,math,time,urllib.request
import numpy as np
from scipy.spatial import ConvexHull

D=8
POINTS=np.array(list(itertools.product((-1,0,1),repeat=D)),dtype=np.int16)
FULL=(1<<len(POINTS))-1
RAW='https://raw.githubusercontent.com/GorrieXIV/Magma/master/libs/data/polytopes/smoothfano8/block{}'

def digits(n,b):
    out=[]
    while n:
        out.append(n%b); n//=b
    return out

def decode(line,base):
    a=digits(int(line),base)
    if len(a)<2 or a[0]!=D: raise ValueError(('bad dimension',a[:3]))
    shift=a[1]; c=[z-shift for z in a[2:]]
    if len(c)%D: raise ValueError(('bad packed length',len(c)))
    return np.array([c[i:i+D] for i in range(0,len(c),D)],dtype=int)

def bitmask(ok):
    return int.from_bytes(np.packbits(ok,bitorder='little').tobytes(),'little') & FULL

def check_neat(U,maxbox):
    m,n=U.shape
    if n!=D: return {'error':'dimension'}
    if not np.array_equal(U[:D],np.eye(D,dtype=int)):
        return {'error':'not_standard','first':U[:D].tolist()}
    # Translation quotient: set the support displacements of the standard
    # unimodular facet to zero. At its dual vertex (1,...,1), strict fan
    # preservation for both +/-b gives |b_j| <= -sum(U_j).
    widths=(-U[D:].sum(axis=1)).astype(int)
    if np.any(widths<0): return {'error':'negative_width','widths':widths.tolist()}
    box=1
    for s in widths: box*=2*int(s)+1
    if box==1: return {'box':1,'valid':1,'neat':True,'rigid':True}
    if box>maxbox: return {'box':box,'unresolved_big':True,'widths':widths.tolist()}
    ranges=[range(-int(s),int(s)+1) for s in widths]
    Z=np.array(list(itertools.product(*ranges)),dtype=np.int16)
    alive=np.ones(len(Z),dtype=bool)
    try:
        hull=ConvexHull(U.astype(float),qhull_options='Qt')
    except Exception as e:
        return {'box':box,'error':'qhull','detail':str(e)}
    simplices=np.unique(np.sort(hull.simplices,axis=1),axis=0)
    # Facets involving more displaced rays usually prune fastest.
    simplices=sorted(simplices,key=lambda I:-sum(int(i)>=D for i in I))
    processed=0
    for I0 in simplices:
        if int(alive.sum())==1: break
        I=np.asarray(I0,dtype=int); M=U[I]
        det=int(round(np.linalg.det(M)))
        if abs(det)!=1:
            return {'box':box,'error':'nonunimodular_hull_facet','indices':I.tolist(),'det':det}
        inv=np.rint(np.linalg.inv(M)).astype(int)
        if not np.array_equal(M@inv,np.eye(D,dtype=int)):
            return {'box':box,'error':'inverse_rounding','indices':I.tolist()}
        alpha=U@inv
        delta=1-alpha.sum(axis=1)
        noninc=np.ones(m,dtype=bool); noninc[I]=False
        js=np.flatnonzero(noninc)
        if np.any(delta[js]<1):
            return {'box':box,'error':'invalid_facet','indices':I.tolist(),'mindelta':int(delta[js].min())}
        r=m-D
        K=np.zeros((len(js),r),dtype=int)
        for t,idx in enumerate(I):
            if idx>=D: K[:,idx-D]+=alpha[js,t]
        for row,j in enumerate(js):
            if j>=D: K[row,j-D]-=1
        B=delta[js]-1
        ids=np.flatnonzero(alive)
        good=np.all(np.abs(Z[ids]@K.T)<=B,axis=1)
        alive[ids[~good]]=False
        processed+=1
    survivors=Z[alive]
    if len(survivors)==1:
        if np.any(survivors[0]): return {'box':box,'error':'zero_missing'}
        return {'box':box,'valid':1,'neat':True,'processed_facets':processed,'fan_facets':len(simplices)}
    # If a nonzero displacement survives every normal-fan inequality, test
    # Q_b cap (-Q_-b).  Because b=0 on the first D normals e_i, every lattice
    # point in this intersection lies in {-1,0,1}^D, so this enumeration is complete.
    extra=U[D:]
    vals=POINTS@extra.T
    masks=[]
    for k,s in enumerate(widths):
        masks.append({q:bitmask(np.abs(vals[:,k]-q)<=1) for q in range(-int(s),int(s)+1)})
    min_inter=len(POINTS); best=None
    for z in survivors:
        if not np.any(z): continue
        mask=FULL
        for k,q in enumerate(z.tolist()):
            mask &= masks[k][q]
            if mask==0: break
        cnt=mask.bit_count()
        if cnt<min_inter: min_inter=cnt; best=z.copy()
        if cnt==0:
            b=[0]*D+[int(q) for q in z]
            return {'box':box,'valid':int(len(survivors)),'neat':False,'b':b,
                    'widths':widths.tolist(),'processed_facets':processed,'fan_facets':len(simplices)}
    return {'box':box,'valid':int(len(survivors)),'neat':True,'min_inter':int(min_inter),
            'best_b':[0]*D+([int(q) for q in best] if best is not None else [0]*(m-D)),
            'processed_facets':processed,'fan_facets':len(simplices)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--block',type=int,required=True);ap.add_argument('--out',required=True);ap.add_argument('--maxbox',type=int,default=200000);a=ap.parse_args()
    with urllib.request.urlopen(RAW.format(a.block),timeout=120) as f: lines=f.read().decode().strip().splitlines()
    base=int(lines[0]); start=a.block*7498+1; scanned=0; nonrigid=0; nonneat=[]; unresolved=[]; errors=[]; maxbox=0; t=time.time()
    for off,line in enumerate(lines[1:]):
        pid=start+off; U=decode(line,base); scanned+=1
        res=check_neat(U,a.maxbox); maxbox=max(maxbox,int(res.get('box',0)))
        if res.get('box',1)>1: nonrigid+=1
        if res.get('neat') is False:
            nonneat.append({'id':pid,'rays':U.tolist(),**res})
        elif res.get('unresolved_big'):
            unresolved.append({'id':pid,'rays':U.tolist(),**res})
        elif 'error' in res:
            errors.append({'id':pid,'rays':U.tolist(),**res})
        if nonneat:
            # A single exact non-neat fiber is enough; preserve it immediately.
            break
    out={'block':a.block,'scanned':scanned,'nonrigid':nonrigid,'nonneat':nonneat,'unresolved':unresolved,'errors':errors,'max_box_seen':maxbox,'elapsed':time.time()-t}
    with open(a.out,'w') as f: json.dump(out,f,separators=(',',':'))
    print('FINAL '+json.dumps({'block':a.block,'scanned':scanned,'nonrigid':nonrigid,'nonneat':len(nonneat),'unresolved':len(unresolved),'errors':len(errors),'maxbox':maxbox,'elapsed':out['elapsed']}),flush=True)
if __name__=='__main__':main()
