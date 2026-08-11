import argparse,gzip,itertools,json,time,urllib.request
import numpy as np
from scipy.spatial import ConvexHull

D=9
POINTS=np.array(list(itertools.product((-1,0,1),repeat=D)),dtype=np.int16)
FULL=(1<<len(POINTS))-1
BASE='https://polymake.org/polytopes/paffenholz/data/smooth-fano/d9/fv-09-{}p.gz'

def bitmask(ok):
    return int.from_bytes(np.packbits(ok,bitorder='little').tobytes(),'little') & FULL

def parse(url,expected):
    with urllib.request.urlopen(url,timeout=180) as raw:
        with gzip.GzipFile(fileobj=raw) as gz:
            cur=[]
            for bb in gz:
                s=bb.decode('ascii').strip()
                if not s: continue
                if s=='FACETS':
                    if cur:
                        if len(cur)!=expected: raise ValueError(('facet count',len(cur),expected))
                        yield np.array(cur,dtype=int);cur=[]
                    continue
                q=[int(z) for z in s.split()]
                if len(q)!=D+1 or q[0]!=1: raise ValueError(('bad row',q))
                # File row 1+a.x >= 0 corresponds to monotone facet normal -a.
                cur.append([-z for z in q[1:]])
            if cur:
                if len(cur)!=expected: raise ValueError(('facet count',len(cur),expected))
                yield np.array(cur,dtype=int)

def check_neat(U,maxbox):
    m,n=U.shape
    if n!=D:return {'error':'dimension'}
    if not np.array_equal(U[:D],np.eye(D,dtype=int)):
        return {'error':'not_standard','first':U[:D].tolist()}
    # Quotient translations by fixing displacement on the standard unimodular facet.
    # Strict preservation of that vertex for both +/-b gives |b_j|<=-sum(U_j).
    widths=(-U[D:].sum(axis=1)).astype(int)
    if np.any(widths<0):return {'error':'negative_width','widths':widths.tolist()}
    box=1
    for s in widths:box*=2*int(s)+1
    if box==1:return {'box':1,'valid':1,'neat':True,'rigid':True}
    if box>maxbox:return {'box':box,'unresolved_big':True,'widths':widths.tolist()}
    ranges=[range(-int(s),int(s)+1) for s in widths]
    Z=np.array(list(itertools.product(*ranges)),dtype=np.int16)
    alive=np.ones(len(Z),dtype=bool)
    try:
        hull=ConvexHull(U.astype(float),qhull_options='Qt')
    except Exception as e:
        return {'box':box,'error':'qhull','detail':str(e)}
    simplices=np.unique(np.sort(hull.simplices,axis=1),axis=0)
    simplices=sorted(simplices,key=lambda I:-sum(int(i)>=D for i in I))
    processed=0
    for I0 in simplices:
        if int(alive.sum())==1:break
        I=np.asarray(I0,dtype=int);M=U[I]
        det=int(round(np.linalg.det(M)))
        if abs(det)!=1:return {'box':box,'error':'nonunimodular_hull_facet','indices':I.tolist(),'det':det}
        inv=np.rint(np.linalg.inv(M)).astype(int)
        if not np.array_equal(M@inv,np.eye(D,dtype=int)):
            return {'box':box,'error':'inverse_rounding','indices':I.tolist()}
        alpha=U@inv
        delta=1-alpha.sum(axis=1)
        noninc=np.ones(m,dtype=bool);noninc[I]=False
        js=np.flatnonzero(noninc)
        if np.any(delta[js]<1):return {'box':box,'error':'invalid_facet','indices':I.tolist(),'mindelta':int(delta[js].min())}
        r=m-D
        K=np.zeros((len(js),r),dtype=int)
        for t,idx in enumerate(I):
            if idx>=D:K[:,idx-D]+=alpha[js,t]
        for row,j in enumerate(js):
            if j>=D:K[row,j-D]-=1
        B=delta[js]-1
        ids=np.flatnonzero(alive)
        good=np.all(np.abs(Z[ids]@K.T)<=B,axis=1)
        alive[ids[~good]]=False
        processed+=1
    survivors=Z[alive]
    if len(survivors)==1:
        if np.any(survivors[0]):return {'box':box,'error':'zero_missing'}
        return {'box':box,'valid':1,'neat':True,'processed_facets':processed,'fan_facets':len(simplices)}
    # For each surviving nonzero displacement, test Q_b cap (-Q_-b).
    # Standard normals e_i have zero displacement, so every lattice point is in {-1,0,1}^D.
    extra=U[D:];vals=POINTS@extra.T;masks=[]
    for k,s in enumerate(widths):
        masks.append({q:bitmask(np.abs(vals[:,k]-q)<=1) for q in range(-int(s),int(s)+1)})
    min_inter=len(POINTS);best=None
    for z in survivors:
        if not np.any(z):continue
        mask=FULL
        for k,q in enumerate(z.tolist()):
            mask &= masks[k][q]
            if mask==0:break
        cnt=mask.bit_count()
        if cnt<min_inter:min_inter=cnt;best=z.copy()
        if cnt==0:
            b=[0]*D+[int(q) for q in z]
            return {'box':box,'valid':int(len(survivors)),'neat':False,'b':b,'widths':widths.tolist(),'processed_facets':processed,'fan_facets':len(simplices)}
    return {'box':box,'valid':int(len(survivors)),'neat':True,'min_inter':int(min_inter),'best_b':[0]*D+([int(q) for q in best] if best is not None else [0]*(m-D)),'processed_facets':processed,'fan_facets':len(simplices)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--facets',type=int,required=True);ap.add_argument('--shard',type=int,required=True);ap.add_argument('--shards',type=int,required=True);ap.add_argument('--out',required=True);ap.add_argument('--maxbox',type=int,default=200000);a=ap.parse_args()
    url=BASE.format(a.facets);scanned=selected=nonrigid=0;nonneat=[];unresolved=[];errors=[];maxseen=0;t=time.time()
    for idx,U in enumerate(parse(url,a.facets),1):
        scanned+=1
        if (idx-1)%a.shards!=a.shard:continue
        selected+=1;res=check_neat(U,a.maxbox);maxseen=max(maxseen,int(res.get('box',0)))
        if res.get('box',1)>1:nonrigid+=1
        rec={'index_in_file':idx,'rays':U.tolist(),**res}
        if res.get('neat') is False:
            nonneat.append(rec);print('NONNEAT '+json.dumps(rec,separators=(',',':')),flush=True);break
        elif res.get('unresolved_big'):unresolved.append(rec)
        elif 'error' in res:errors.append(rec)
        if selected%10000==0:print(json.dumps({'facets':a.facets,'shard':a.shard,'selected':selected,'nonneat':len(nonneat),'unresolved':len(unresolved),'errors':len(errors),'sec':round(time.time()-t,1)}),flush=True)
    out={'dimension':D,'facet_count':a.facets,'shard':a.shard,'shards':a.shards,'scanned_file':scanned,'selected':selected,'nonrigid':nonrigid,'nonneat':nonneat,'unresolved':unresolved,'errors':errors,'max_box_seen':maxseen,'elapsed':time.time()-t}
    with open(a.out,'w') as f:json.dump(out,f,separators=(',',':'))
    print('FINAL '+json.dumps({k:v for k,v in out.items() if k not in ('nonneat','unresolved','errors') }|{'nonneat_count':len(nonneat),'unresolved_count':len(unresolved),'error_count':len(errors)},separators=(',',':')),flush=True)
if __name__=='__main__':main()
