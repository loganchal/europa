import argparse,gzip,itertools,json,time,urllib.request,collections
import numpy as np
from scipy.spatial import ConvexHull

D=9
POINTS=np.array(list(itertools.product((-1,0,1),repeat=D)),dtype=np.int16)
FULL=(1<<len(POINTS))-1
URL='https://polymake.org/polytopes/paffenholz/data/smooth-fano/d9/fv-09-{:02d}p.gz'

def bitmask(ok):
    return int.from_bytes(np.packbits(ok,bitorder='little').tobytes(),'little') & FULL

def polytopes(path):
    cur=[]
    with gzip.open(path,'rt') as f:
        for raw in f:
            s=raw.strip()
            if s=='FACETS':
                if cur:
                    yield np.array(cur,dtype=np.int64);cur=[]
                continue
            if not s:continue
            v=[int(x) for x in s.split()]
            if len(v)!=D+1 or v[0]!=1:raise ValueError(('bad row',s))
            cur.append([-q for q in v[1:]])
    if cur:yield np.array(cur,dtype=np.int64)

def exact_inverse_unimodular(M):
    try:
        R=np.rint(np.linalg.inv(M.astype(float))).astype(np.int64)
    except Exception:
        return None
    I=np.eye(D,dtype=np.int64)
    if not np.array_equal(M@R,I):return None
    if not np.array_equal(R@M,I):return None
    return R

def certify_facets(U,simplices):
    facets=[tuple(map(int,I)) for I in simplices]
    if not facets:return False,{'error':'no_facets'}
    if len(set(facets))!=len(facets):return False,{'error':'duplicate_facets'}
    m=len(U);ones=np.ones(D,dtype=np.int64);used=set();ridge_map=collections.defaultdict(list)
    for fi,I in enumerate(facets):
        if len(I)!=D or len(set(I))!=D:return False,{'error':'bad_simplex','indices':list(I)}
        M=U[list(I)];inv=exact_inverse_unimodular(M)
        if inv is None:return False,{'error':'nonunimodular_or_singular','indices':list(I)}
        x=inv@ones
        vals=U@x
        if int(vals.max())>1:return False,{'error':'not_supporting','indices':list(I),'max':int(vals.max())}
        eq=tuple(np.flatnonzero(vals==1).tolist())
        if eq!=tuple(sorted(I)):
            return False,{'error':'incidence_mismatch','indices':list(I),'equal':list(eq)}
        used.update(I)
        for ridge in itertools.combinations(sorted(I),D-1):ridge_map[ridge].append(fi)
    if used!=set(range(m)):
        return False,{'error':'unused_rays','unused':sorted(set(range(m))-used)}
    bad=[(r,v) for r,v in ridge_map.items() if len(v)!=2]
    if bad:
        r,v=bad[0];return False,{'error':'ridge_incidence','ridge':list(r),'count':len(v)}
    adj=[set() for _ in facets]
    for fs in ridge_map.values():
        a,b=fs;adj[a].add(b);adj[b].add(a)
    seen={0};stack=[0]
    while stack:
        a=stack.pop()
        for b in adj[a]:
            if b not in seen:seen.add(b);stack.append(b)
    if len(seen)!=len(facets):return False,{'error':'facet_graph_disconnected','seen':len(seen),'total':len(facets)}
    return True,{'facet_count':len(facets),'ridge_count':len(ridge_map)}

def check_neat(U,maxbox):
    m,n=U.shape
    if n!=D:return {'error':'dimension'}
    if not np.array_equal(U[:D],np.eye(D,dtype=np.int64)):
        return {'error':'not_standard','first':U[:D].tolist()}
    widths=(-U[D:].sum(axis=1)).astype(int)
    if np.any(widths<0):return {'error':'negative_width','widths':widths.tolist()}
    box=1
    for s in widths:box*=2*int(s)+1
    if box>maxbox:return {'box':box,'unresolved_big':True,'widths':widths.tolist()}
    try:hull=ConvexHull(U.astype(float),qhull_options='Qt')
    except Exception as e:return {'box':box,'error':'qhull','detail':str(e)}
    simplices=np.unique(np.sort(hull.simplices,axis=1),axis=0)
    ok,cert=certify_facets(U,simplices)
    if not ok:return {'box':box,**cert}
    if box==1:return {'box':1,'valid':1,'neat':True,'rigid':True,'fan_certificate':cert}
    Z=np.array(list(itertools.product(*[range(-int(s),int(s)+1) for s in widths])),dtype=np.int16)
    alive=np.ones(len(Z),dtype=bool)
    simplices=sorted(simplices,key=lambda I:-sum(int(i)>=D for i in I))
    processed=0
    for I0 in simplices:
        if int(alive.sum())==1:break
        I=np.asarray(I0,dtype=int);M=U[I];inv=exact_inverse_unimodular(M)
        if inv is None:return {'box':box,'error':'inverse_after_certificate','indices':I.tolist()}
        alpha=U@inv
        delta=1-alpha.sum(axis=1)
        noninc=np.ones(m,dtype=bool);noninc[I]=False;js=np.flatnonzero(noninc)
        if np.any(delta[js]<1):return {'box':box,'error':'invalid_slack','indices':I.tolist(),'mindelta':int(delta[js].min())}
        K=np.zeros((len(js),m-D),dtype=np.int64)
        for t,idx in enumerate(I):
            if idx>=D:K[:,idx-D]+=alpha[js,t]
        for row,j in enumerate(js):
            if j>=D:K[row,j-D]-=1
        B=delta[js]-1
        ids=np.flatnonzero(alive)
        good=np.all(np.abs(Z[ids].astype(np.int64)@K.T)<=B,axis=1)
        alive[ids[~good]]=False;processed+=1
    survivors=Z[alive]
    if len(survivors)==0:return {'box':box,'error':'zero_displacement_removed'}
    zero_rows=np.flatnonzero(np.all(survivors==0,axis=1))
    if len(zero_rows)!=1:return {'box':box,'error':'zero_displacement_count','count':int(len(zero_rows))}
    if len(survivors)==1:
        return {'box':box,'valid':1,'neat':True,'processed_facets':processed,'fan_certificate':cert}
    extra=U[D:];vals=POINTS@extra.T;masks=[]
    for k,s in enumerate(widths):
        masks.append({q:bitmask(np.abs(vals[:,k]-q)<=1) for q in range(-int(s),int(s)+1)})
    min_inter=len(POINTS);best=None
    for z in survivors:
        if not np.any(z):continue
        mask=FULL
        for k,q in enumerate(z.tolist()):
            mask &= masks[k][int(q)]
            if not mask:break
        cnt=mask.bit_count()
        if cnt<min_inter:min_inter=cnt;best=z.copy()
        if cnt==0:
            return {'box':box,'valid':int(len(survivors)),'neat':False,'b':[0]*D+[int(q) for q in z],
                    'widths':widths.tolist(),'processed_facets':processed,'fan_certificate':cert}
    return {'box':box,'valid':int(len(survivors)),'neat':True,'min_inter':int(min_inter),
            'best_b':[0]*D+([int(q) for q in best] if best is not None else [0]*(m-D)),
            'processed_facets':processed,'fan_certificate':cert}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--facets',type=int,required=True);ap.add_argument('--shard',type=int,required=True);ap.add_argument('--shards',type=int,required=True);ap.add_argument('--out',required=True);ap.add_argument('--maxbox',type=int,default=200000);a=ap.parse_args()
    path=f'/tmp/fv-09-{a.facets}p.gz';urllib.request.urlretrieve(URL.format(a.facets),path)
    scanned=selected=0;nonneat=[];unresolved=[];errors=[];maxbox=0;best_inter=(10**9,None,None);t=time.time()
    for ordinal,U in enumerate(polytopes(path),1):
        scanned+=1
        if (ordinal-1)%a.shards!=a.shard:continue
        selected+=1;res=check_neat(U,a.maxbox);maxbox=max(maxbox,int(res.get('box',0)))
        if res.get('min_inter',10**9)<best_inter[0]:best_inter=(int(res['min_inter']),ordinal,res.get('best_b'))
        rec={'ordinal':ordinal,'rays':U.tolist(),**res}
        if res.get('neat') is False:
            nonneat.append(rec);print('NONNEAT '+json.dumps(rec,separators=(',',':')),flush=True);break
        if res.get('unresolved_big'):unresolved.append(rec)
        elif 'error' in res:errors.append(rec)
        if selected%10000==0:print(json.dumps({'facets':a.facets,'shard':a.shard,'selected':selected,'nonneat':len(nonneat),'unresolved':len(unresolved),'errors':len(errors),'best_inter':best_inter[0],'sec':round(time.time()-t,1)}),flush=True)
    out={'dimension':D,'facets':a.facets,'shard':a.shard,'shards':a.shards,'file_scanned':scanned,'selected':selected,'nonneat':nonneat,'unresolved':unresolved,'errors':errors,'max_box_seen':maxbox,'best_inter':best_inter,'elapsed':time.time()-t}
    with open(a.out,'w') as f:json.dump(out,f,separators=(',',':'))
    print('FINAL '+json.dumps({'facets':a.facets,'shard':a.shard,'shards':a.shards,'selected':selected,'nonneat':len(nonneat),'unresolved':len(unresolved),'errors':len(errors),'maxbox':maxbox,'best_inter':best_inter[0],'elapsed':out['elapsed']}),flush=True)
if __name__=='__main__':main()
