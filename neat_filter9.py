import argparse,gzip,itertools,json,os,tempfile,time,urllib.request
import numpy as np

D=9
POINTS=np.array(list(itertools.product((-1,0,1),repeat=D)),dtype=np.int16)
FULL=(1<<len(POINTS))-1
URL='https://polymake.org/polytopes/paffenholz/data/smooth-fano/d9/fv-09-{:02d}p.gz'
MASK_CACHE={}

def bitmask(ok):
    return int.from_bytes(np.packbits(ok,bitorder='little').tobytes(),'little') & FULL

def get_mask(u,q):
    key=(tuple(map(int,u)),int(q))
    m=MASK_CACHE.get(key)
    if m is None:
        vals=POINTS@np.asarray(u,dtype=np.int16)
        m=bitmask(np.abs(vals-int(q))<=1)
        MASK_CACHE[key]=m
    return m

def records(path):
    cur=[]
    with gzip.open(path,'rt') as f:
        for line in f:
            s=line.strip()
            if s=='FACETS':
                if cur:
                    yield np.array(cur,dtype=int);cur=[]
                continue
            if not s: continue
            z=[int(x) for x in s.split()]
            if len(z)==D+1 and z[0]==1:
                # File stores 1 + a.x >= 0; use U=-a so U.x <= 1.
                cur.append([-x for x in z[1:]])
        if cur: yield np.array(cur,dtype=int)

def widths_for(U):
    if U.shape[1]!=D:return None
    if U.shape[0]<D or not np.array_equal(U[:D],np.eye(D,dtype=int)):return None
    return (-U[D:].sum(axis=1)).astype(int)

def intersection_mask(U,qs):
    m=FULL
    for u,q in zip(U[D:],qs):
        m &= get_mask(u,int(q))
        if m==0:break
    return m

def exact_fan_preserved(U,qs):
    # Expensive exact-on-return verification, used only for empty-intersection candidates.
    from scipy.spatial import ConvexHull
    b=np.array([0]*D+list(map(int,qs)),dtype=int)
    try:
        hull=ConvexHull(U.astype(float),qhull_options='Qt')
    except Exception as e:
        return False,{'error':'qhull','detail':str(e)}
    facets=np.unique(np.sort(hull.simplices,axis=1),axis=0)
    if len(facets)==0:return False,{'error':'no_facets'}
    for I0 in facets:
        I=np.asarray(I0,dtype=int)
        if len(I)!=D:return False,{'error':'nonsimplicial','indices':I.tolist()}
        M=U[I]
        det=int(round(np.linalg.det(M)))
        if abs(det)!=1:return False,{'error':'nonunimodular','indices':I.tolist(),'det':det}
        inv=np.rint(np.linalg.inv(M)).astype(int)
        if not np.array_equal(M@inv,np.eye(D,dtype=int)):
            return False,{'error':'inverse','indices':I.tolist()}
        alpha=U@inv
        delta=1-alpha.sum(axis=1)
        inc=np.zeros(len(U),dtype=bool);inc[I]=True
        js=np.flatnonzero(~inc)
        if np.any(delta[js]<1):return False,{'error':'bad_slack','indices':I.tolist()}
        lhs=alpha[js]@b[I]-b[js]
        if np.any(np.abs(lhs)>delta[js]-1):
            return False,{'violating_facet':I.tolist(),'max_excess':int(np.max(np.abs(lhs)-(delta[js]-1)))}
    return True,{'fan_facets':int(len(facets))}

def candidate_assignments(U,widths,exhaust_cap=5000,beam_size=64,max_candidates=8):
    r=len(widths)
    if r==0:return [],1,'rigid'
    box=1
    for w in widths:
        if w<0:return [],0,'negative_width'
        box*=2*int(w)+1
    if box==1:return [],box,'rigid'
    choices=[list(range(-int(w),int(w)+1)) for w in widths]
    out=[]
    if box<=exhaust_cap:
        for qs in itertools.product(*choices):
            if not any(qs):continue
            if intersection_mask(U,qs)==0:
                out.append(tuple(map(int,qs)))
                if len(out)>=max_candidates:break
        return out,box,'exhaustive'
    # Beam search: process most restrictive extra normals first, retain low-cardinality intersections.
    order=[]
    for k,(u,ch) in enumerate(zip(U[D:],choices)):
        best=min(get_mask(u,q).bit_count() for q in ch)
        order.append((best,k))
    order=[k for _,k in sorted(order)]
    states=[(FULL,{})]
    seen_empty=set()
    for k in order:
        nxt={}
        for mask,asg in states:
            for q in choices[k]:
                nm=mask & get_mask(U[D+k],q)
                nasg=dict(asg);nasg[k]=int(q)
                if nm==0:
                    qs=tuple(nasg.get(j,0) for j in range(r))
                    # Complete unassigned entries with zero, which preserves emptiness.
                    if any(qs) and qs not in seen_empty:
                        seen_empty.add(qs);out.append(qs)
                        if len(out)>=max_candidates:return out,box,'beam'
                    continue
                old=nxt.get(nm)
                if old is None:nxt[nm]=nasg
        if not nxt:break
        ranked=sorted(nxt.items(),key=lambda kv:(kv[0].bit_count(),len(kv[1])))[:beam_size]
        states=[(m,a) for m,a in ranked]
    return out,box,'beam'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--facets',type=int,required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--exhaust-cap',type=int,default=5000);ap.add_argument('--beam',type=int,default=64);a=ap.parse_args()
    url=URL.format(a.facets);tmp=tempfile.NamedTemporaryFile(suffix='.gz',delete=False);tmp.close()
    urllib.request.urlretrieve(url,tmp.name)
    t=time.time();scanned=nonrigid=0;candidate_polys=verified_false=0;errors=[];witness=None;maxbox=1;mode_counts={};box_bins={'1':0,'2-99':0,'100-4999':0,'5000+':0}
    for idx,U in enumerate(records(tmp.name),start=1):
        scanned+=1
        w=widths_for(U)
        if w is None:
            errors.append({'index':idx,'error':'not_standard','first':U[:D].tolist()});continue
        cands,box,mode=candidate_assignments(U,w,a.exhaust_cap,a.beam)
        maxbox=max(maxbox,int(box));mode_counts[mode]=mode_counts.get(mode,0)+1
        if box==1:box_bins['1']+=1
        elif box<100:box_bins['2-99']+=1
        elif box<5000:box_bins['100-4999']+=1
        else:box_bins['5000+']+=1
        if box>1:nonrigid+=1
        if not cands:continue
        candidate_polys+=1
        for qs in cands:
            ok,detail=exact_fan_preserved(U,qs)
            if ok:
                witness={'facet_count':a.facets,'index_in_file':idx,'normals':U.tolist(),'b':[0]*D+list(qs),'widths':w.tolist(),'box':int(box),'verification':detail}
                print('WITNESS '+json.dumps(witness,separators=(',',':')),flush=True)
                break
            verified_false+=1
        if witness:break
        if scanned%200000==0:
            print(json.dumps({'facets':a.facets,'scanned':scanned,'candidates':candidate_polys,'false':verified_false,'maxbox':maxbox,'cache':len(MASK_CACHE),'sec':round(time.time()-t,1)}),flush=True)
    os.unlink(tmp.name)
    out={'dimension':D,'facet_count':a.facets,'scanned':scanned,'nonrigid':nonrigid,'candidate_polys':candidate_polys,'verified_false_candidates':verified_false,'witness':witness,'errors':errors[:20],'error_count':len(errors),'max_box':maxbox,'box_bins':box_bins,'mode_counts':mode_counts,'mask_cache':len(MASK_CACHE),'elapsed_sec':time.time()-t}
    with open(a.out,'w') as f:json.dump(out,f,separators=(',',':'))
    print('FINAL '+json.dumps({k:v for k,v in out.items() if k not in ('witness','errors') }|{'has_witness':witness is not None},separators=(',',':')),flush=True)

if __name__=='__main__':main()
