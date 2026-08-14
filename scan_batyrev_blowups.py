import argparse,itertools,json,math,time
from functools import lru_cache
import numpy as np

D=10
PTS=np.array(list(itertools.product((-1,0,1),repeat=D)),dtype=np.int16)
FULL=(1<<len(PTS))-1
MASK_CACHE={}

def bitmask(ok):
    return int.from_bytes(np.packbits(ok,bitorder='little').tobytes(),'little') & FULL

def shifted_mask(u,q):
    k=(tuple(map(int,u)),int(q));m=MASK_CACHE.get(k)
    if m is None:
        vals=PTS@np.asarray(u,dtype=np.int16)
        m=bitmask(np.abs(vals-int(q))<=1);MASK_CACHE[k]=m
    return m

def invu(M):
    M=np.asarray(M,dtype=int)
    try:X=np.rint(np.linalg.inv(M.astype(float))).astype(int)
    except Exception:return None
    return X if np.array_equal(M@X,np.eye(D,dtype=int)) else None

def pos_comps(n,k):
    for cuts in itertools.combinations(range(1,n),k-1):
        prev=0;out=[]
        for c in cuts+(n,):out.append(c-prev);prev=c
        yield tuple(out)

def nondec_tuples(length,maxsum):
    if length==0:yield ();return
    def rec(pref,last,rem):
        if len(pref)==length:yield tuple(pref);return
        slots=length-len(pref)
        for x in range(last,rem//slots+1):yield from rec(pref+[x],x,rem-x)
    yield from rec([],0,maxsum)

def all_cases():
    out=[]
    for p in pos_comps(13,5):
        p0,p1,p2,p3,p4=p
        if p1+p2-p4<=0 or p3+p4-p1<=0:continue
        M=min(p0+p1-p3-1,p4+p0-1)
        if M<0:continue
        for c in nondec_tuples(p2-1,M):
            for d in nondec_tuples(p3,M-sum(c)):
                out.append((p,c,d))
    assert len(out)==1510
    return out

CASES=all_cases()

def case_data(p,c,d):
    p0,p1,p2,p3,p4=p
    starts=np.cumsum([0,p0,p1,p2,p3,p4])
    V=list(range(starts[0],starts[1]));Y=list(range(starts[1],starts[2]));Z=list(range(starts[2],starts[3]));T=list(range(starts[3],starts[4]));U=list(range(starts[4],starts[5]))
    pcs=[set(V+Y),set(Y+Z),set(Z+T),set(T+U),set(U+V)]
    cones=[];allidx=set(range(13))
    for O in itertools.combinations(range(13),3):
        rem=allidx-set(O)
        if all(not pc.issubset(rem) for pc in pcs):cones.append(tuple(sorted(rem)))
    A=np.zeros((5,13),dtype=int)
    A[0,V]=1;A[0,Y]=1
    for jj,coef in enumerate(c,start=1):A[0,Z[jj]]=-coef
    for jj,coef in enumerate(d):A[0,T[jj]]=-(coef+1)
    A[1,Y]=1;A[1,Z]=1;A[1,U]=-1
    A[2,Z]=1;A[2,T]=1
    A[3,T]=1;A[3,U]=1;A[3,Y]=-1
    A[4,U]=1;A[4,V]=1
    for jj,coef in enumerate(c,start=1):A[4,Z[jj]]=-coef
    for jj,coef in enumerate(d):A[4,T[jj]]=-coef
    return pcs,cones,A

def reconstruct(p,c,d):
    pcs,cones,A=case_data(p,c,d)
    for B in cones:
        O=tuple(i for i in range(13) if i not in set(B));AO=A[:,O];rows=None
        for rr in itertools.combinations(range(5),3):
            sub=AO[list(rr),:]
            if round(np.linalg.det(sub))!=0:rows=rr;break
        if rows is None:continue
        sub=AO[list(rows),:];rays=[None]*13
        for j,idx in enumerate(B):
            e=np.zeros(D,dtype=int);e[j]=1;rays[idx]=e
        ok=True
        for j in range(D):
            rhs=-A[list(rows),B[j]]
            try:xf=np.linalg.solve(sub.astype(float),rhs.astype(float))
            except Exception:ok=False;break
            x=np.rint(xf).astype(int)
            if not np.array_equal(sub@x,rhs) or not np.array_equal(AO@x,-A[:,B[j]]):ok=False;break
            for k,idx in enumerate(O):
                if rays[idx] is None:rays[idx]=np.zeros(D,dtype=int)
                rays[idx][j]=x[k]
        if not ok:continue
        R0=np.array(rays,dtype=int)
        if not np.array_equal(A@R0,np.zeros((5,D),dtype=int)):continue
        order=list(B)+list(O);invorder={old:new for new,old in enumerate(order)}
        R=R0[order];C=[tuple(sorted(invorder[i] for i in cone)) for cone in cones]
        if not np.array_equal(R[:D],np.eye(D,dtype=int)):raise RuntimeError('basis')
        return R,C
    return None

@lru_cache(None)
def faces_cached(Ckey):
    fs=set()
    for I in Ckey:
        for k in range(2,D+1):fs.update(itertools.combinations(I,k))
    return tuple(sorted(fs,key=lambda s:(len(s),s)))

def cone_vertex(R,I):
    X=invu(R[list(I)])
    return None if X is None else X@np.ones(D,dtype=int)

def valid_blowups(R,C):
    C=tuple(tuple(I) for I in C);csets=[set(I) for I in C]
    verts=np.array([cone_vertex(R,I) for I in C],dtype=int);Aeval=R@verts.T
    out=[]
    for S in faces_cached(C):
        Ss=set(S);contain=np.array([Ss.issubset(cs) for cs in csets],dtype=bool)
        sums=Aeval[list(S),:].sum(axis=0)
        if np.any(sums[~contain]>0):continue
        w=R[list(S)].sum(axis=0);T=np.vstack([R,w]);wi=len(R);ok=True
        for ci,I in enumerate(C):
            if not contain[ci]:continue
            Iset=set(I)
            for s in S:
                J=tuple(sorted((Iset-{s})|{wi}));m=cone_vertex(T,J)
                if m is None:ok=False;break
                dots=T@m;jset=set(J)
                for j,z in enumerate(dots):
                    if j in jset:
                        if z!=1:ok=False;break
                    elif z>=1:ok=False;break
                if not ok:break
            if not ok:break
        if ok:out.append((S,tuple(map(int,w))))
    return out

def star_subdivide(R,C,S):
    S=set(S);w=R[list(S)].sum(axis=0);T=np.vstack([R,w]);wi=len(R);new=[]
    for I in C:
        Is=set(I)
        if S.issubset(Is):
            for s in S:new.append(tuple(sorted((Is-{s})|{wi})))
        else:new.append(tuple(I))
    return T,list(dict.fromkeys(new))

def standardize(R,C):
    B=tuple(C[0]);X=invu(R[list(B)])
    if X is None:return None
    Rt=R@X;bs=set(B);order=list(B)+[i for i in range(len(R)) if i not in bs];mp={old:new for new,old in enumerate(order)}
    Rs=Rt[order];Cs=[tuple(sorted(mp[i] for i in I)) for I in C]
    if not np.array_equal(Rs[:D],np.eye(D,dtype=int)):return None
    return Rs,Cs

def fan_constraints(R,C):
    m=len(R)-D;cons={}
    for I0 in C:
        I=list(I0);X=invu(R[I])
        if X is None:return None
        alpha=R@X;delta=1-alpha.sum(axis=1);inc=set(I)
        for j in range(len(R)):
            if j in inc:continue
            if delta[j]<1:return None
            coeff=np.zeros(m,dtype=int)
            for pos,ri in enumerate(I):
                if ri>=D:coeff[ri-D]+=alpha[j,pos]
            if j>=D:coeff[j-D]-=1
            bound=int(delta[j]-1)
            if np.all(coeff==0):
                if bound<0:return None
                continue
            tup=tuple(map(int,coeff));first=next(x for x in tup if x)
            if first<0:tup=tuple(-x for x in tup)
            old=cons.get(tup)
            if old is None or bound<old:cons[tup]=bound
    return tuple((k,v) for k,v in cons.items())

def min_intersection(R,C):
    std=standardize(R,C)
    if std is None:return None
    R,C=std;m=len(R)-D;cons=fan_constraints(R,C)
    if cons is None:return None
    widths=tuple(int(-r.sum()) for r in R[D:])
    if any(w<0 for w in widths):return None
    box=math.prod(2*w+1 for w in widths);valid=0;best=(10**9,None)
    opts=[]
    for r,w in zip(R[D:],widths):opts.append({q:shifted_mask(r,q) for q in range(-w,w+1)})
    for qs in itertools.product(*[range(-w,w+1) for w in widths]):
        good=True
        for a,b in cons:
            z=sum(ai*qi for ai,qi in zip(a,qs))
            if abs(z)>b:good=False;break
        if not good:continue
        valid+=1;mask=FULL
        for op,q in zip(opts,qs):
            mask &= op[q]
            if not mask:break
        n=mask.bit_count()
        if n<best[0]:
            best=(n,tuple(map(int,qs)))
            if n==0:break
    return {'best':best,'valid':valid,'box':box,'widths':widths,'normals':R.tolist() if best[0]==0 else None,'cones':[list(x) for x in C] if best[0]==0 else None}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,required=True);ap.add_argument('--end',type=int,required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    t=time.time();parents=blowups=errors=0;best=(10**9,None);witness=None;maxbox=0;maxvalid=0
    for idx in range(a.start,a.end):
        p,c,d=CASES[idx];rec=reconstruct(p,c,d)
        if rec is None:errors+=1;continue
        R,C=rec;parents+=1;vb=valid_blowups(R,C);blowups+=len(vb)
        for bi,(S,w) in enumerate(vb):
            T,C2=star_subdivide(R,C,S);res=min_intersection(T,C2)
            if res is None:errors+=1;continue
            maxbox=max(maxbox,res['box']);maxvalid=max(maxvalid,res['valid'])
            if res['best'][0]<best[0]:
                best=(res['best'][0],{'parent':idx,'p':p,'c':c,'d':d,'blowup_index':bi,'face':S,'q':res['best'][1],'widths':res['widths']})
                print('BEST',json.dumps(best[1]|{'intersection':best[0]},separators=(',',':')),flush=True)
            if res['best'][0]==0:
                witness=best[1]|{'intersection':0,'normals':res['normals'],'cones':res['cones']};break
        if witness:break
        if (idx-a.start+1)%10==0:print('PROG',json.dumps({'idx':idx,'parents':parents,'blowups':blowups,'best':best[0],'maxbox':maxbox,'maxvalid':maxvalid,'cache':len(MASK_CACHE),'sec':round(time.time()-t,1)}),flush=True)
    out={'start':a.start,'end':a.end,'parents':parents,'blowups':blowups,'errors':errors,'best_inter':best[0],'best':best[1],'witness':witness,'max_box':maxbox,'max_valid':maxvalid,'mask_cache':len(MASK_CACHE),'elapsed':time.time()-t}
    open(a.out,'w').write(json.dumps(out,separators=(',',':')))
    print('FINAL',json.dumps({k:v for k,v in out.items() if k!='witness'}|{'has_witness':witness is not None},separators=(',',':')),flush=True)
    if witness is None:assert parents==a.end-a.start and errors==0,out

if __name__=='__main__':main()
