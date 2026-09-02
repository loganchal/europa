#!/usr/bin/env python3
from __future__ import annotations
import argparse, itertools, json, os, random, time, urllib.request
import numpy as np

D=8
TERNARY=np.array(list(itertools.product((-1,0,1),repeat=D)),dtype=np.int16)
IDENT=np.eye(D,dtype=np.int16)


def decode_line(line:str,base:int):
    line=line.strip()
    if base==10:
        coeff=[ord(ch)-48 for ch in reversed(line)]
    else:
        n=int(line); coeff=[]
        while n:
            n,r=divmod(n,base);coeff.append(r)
    d=coeff[0];shift=coeff[1]
    if d!=D:raise ValueError(f'dimension {d}')
    a=np.fromiter((x-shift for x in coeff[2:]),dtype=np.int16)
    if a.size%d:raise ValueError('bad encoded length')
    U=a.reshape((-1,d))
    if U.shape[0]<d:raise ValueError('too few rays')
    return U


def det_int(A):
    a=[[int(x) for x in row] for row in A]
    n=len(a);sign=1;prev=1
    for k in range(n-1):
        if a[k][k]==0:
            p=next((i for i in range(k+1,n) if a[i][k]),None)
            if p is None:return 0
            a[k],a[p]=a[p],a[k];sign=-sign
        pivot=a[k][k]
        for i in range(k+1,n):
            for j in range(k+1,n):
                a[i][j]=(a[i][j]*pivot-a[i][k]*a[k][j])//prev
        prev=pivot
        for i in range(k+1,n):a[i][k]=0
        for j in range(k+1,n):a[k][j]=0
    return sign*a[-1][-1]


def sign_reps(E):
    out=[]
    for v in E:
        nz=np.flatnonzero(v)
        if len(nz) and v[nz[0]]>0:out.append(v)
    return np.asarray(out,dtype=np.int16)


def exact_if_one(A):
    return abs(det_int(A.tolist() if hasattr(A,'tolist') else A))==1


def random_full_rank(R,rng,attempts=50):
    n=len(R)
    for _ in range(attempts):
        idx=rng.sample(range(n),D)
        A=R[idx].astype(np.int64)
        d0=int(round(np.linalg.det(A)))
        if d0:return A,d0
    chosen=[];rank=0
    for v in R:
        B=np.asarray(chosen+[v.tolist()],dtype=np.float64)
        rr=np.linalg.matrix_rank(B,tol=1e-8)
        if rr>rank:
            chosen.append(v.tolist());rank=rr
            if rank==D:
                A=np.asarray(chosen,dtype=np.int64)
                return A,int(round(np.linalg.det(A)))
    return None,0


def find_basis(E,seed=0,restarts=8,max_steps=16):
    S={tuple(map(int,v)) for v in E}
    if all(tuple(map(int,v)) in S for v in IDENT):
        return IDENT.astype(np.int64)
    R=sign_reps(E)
    if len(R)<D:return None
    rng=random.Random(seed);n=len(R)
    for _ in range(3):
        inds=np.array([[rng.randrange(n) for _ in range(D)] for __ in range(128)],dtype=np.int64)
        B=R[inds].astype(np.float64)
        ds=np.rint(np.linalg.det(B)).astype(np.int64)
        hits=np.flatnonzero(np.abs(ds)==1)
        for h in hits[:3]:
            A=R[inds[h]].astype(np.int64)
            if exact_if_one(A):return A
    for _ in range(restarts):
        A,d0=random_full_rank(R,rng)
        if A is None:return None
        if abs(d0)==1 and exact_if_one(A):return A
        seen=set()
        for step in range(max_steps):
            d0=int(round(np.linalg.det(A)))
            if d0==0:break
            key=tuple(map(tuple,A.tolist()))
            if key in seen:break
            seen.add(key)
            try:adjcols=np.rint(d0*np.linalg.inv(A)).astype(np.int64)
            except np.linalg.LinAlgError:break
            best=None;order=list(range(D));rng.shuffle(order)
            for i in order:
                c=adjcols[:,i]
                vals=R.astype(np.int64)@c
                hit=np.flatnonzero(np.abs(vals)==1)
                if len(hit):
                    for h in hit[:3]:
                        B=A.copy();B[i]=R[int(h)]
                        if exact_if_one(B):return B
                nz=np.flatnonzero(vals)
                if len(nz):
                    av=np.abs(vals[nz]);j=int(nz[int(np.argmin(av))]);score=int(abs(vals[j]))
                    if best is None or score<best[0]:best=(score,i,j)
            if best is None:break
            score,i,j=best
            if score>=abs(d0) and step>1:break
            A[i]=R[j]
            if score==1 and exact_if_one(A):return A
    return None


def rank_mod(E,p):
    A=(E.astype(np.int64)%p).copy();m,n=A.shape;r=0
    for j in range(n):
        rows=np.flatnonzero(A[r:,j])
        if not len(rows):continue
        i=r+int(rows[0]);A[[r,i]]=A[[i,r]]
        inv=pow(int(A[r,j]),-1,p);A[r]=(A[r]*inv)%p
        for i in range(m):
            if i!=r and A[i,j]:A[i]=(A[i]-A[i,j]*A[r])%p
        r+=1
        if r==n:break
    return r


def fetch_block(block,retries=4):
    url=f'https://raw.githubusercontent.com/GorrieXIV/Magma/master/libs/data/polytopes/smoothfano8/block{block}'
    for k in range(retries):
        try:
            with urllib.request.urlopen(url,timeout=60) as f:return f.read().decode('ascii').splitlines()
        except Exception:
            if k+1==retries:raise
            time.sleep(2**k)


def scan(shard,nshards,outdir):
    os.makedirs(outdir,exist_ok=True)
    failures=[];bad_normal_form=[];count=0;min_E=10**9;min_ids=[];start=time.time();block_stats=[]
    for block in range(101):
        if block%nshards!=shard:continue
        lines=fetch_block(block);base=int(lines[0]);bc=0;bmin=10**9
        for local,line in enumerate(lines[1:]):
            poly_id=block*7498+local+1
            if poly_id>749892:break
            U=decode_line(line,base)
            if not np.array_equal(U[:D],IDENT):bad_normal_form.append(poly_id)
            vals=TERNARY@U.T
            E=TERNARY[np.max(np.abs(vals),axis=1)<=1]
            ne=len(E);count+=1;bc+=1;bmin=min(bmin,ne)
            if ne<min_E:min_E=ne;min_ids=[poly_id]
            elif ne==min_E:min_ids.append(poly_id)
            basis=find_basis(E,seed=poly_id)
            if basis is None:
                failures.append({'id':poly_id,'block':block,'local':local,'n_rays':int(len(U)),'n_E':ne,
                                 'rank2':rank_mod(E,2),'rank3':rank_mod(E,3),
                                 'rays':U.astype(int).tolist(),'E':E.astype(int).tolist()})
                print('HEURISTIC_FAILURE',poly_id,'rays',len(U),'E',ne,'r2',failures[-1]['rank2'],'r3',failures[-1]['rank3'],flush=True)
            elif abs(det_int(basis.tolist()))!=1:
                raise AssertionError('nonexact basis accepted')
        block_stats.append({'block':block,'count':bc,'min_E':bmin})
        print('BLOCK',block,'count',bc,'minE',bmin,'failures',len(failures),'elapsed',round(time.time()-start,1),flush=True)
    result={'shard':shard,'nshards':nshards,'count':count,'min_E':min_E,'min_ids':min_ids,
            'heuristic_failures':failures,'bad_normal_form':bad_normal_form,'block_stats':block_stats,
            'elapsed_sec':time.time()-start}
    path=os.path.join(outdir,f'shard-{shard}.json')
    with open(path,'w') as f:json.dump(result,f,separators=(',',':'))
    print('RESULT',json.dumps({k:v for k,v in result.items() if k not in ('heuristic_failures','block_stats')},separators=(',',':')),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--nshards',type=int,default=10);ap.add_argument('--outdir',default='ewald-results')
    a=ap.parse_args();scan(a.shard,a.nshards,a.outdir)
