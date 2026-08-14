import argparse,heapq,itertools,json,time
import numpy as np
import fcomp10 as F

# Fixed coordinates x0,...,x9.  These 10 basis rays are never deleted,
# hence every Ewald point remains in {-1,0,1}^10 and F.emask is complete.
ROOT=[]
for i in range(1,10):
    r=[0]*10;r[i]=1;ROOT.append(r)
ROOT += [[-9]+[-1]*9,[1]+[0]*9,[-1]+[0]*9]
ROOT=np.asarray(ROOT,dtype=int)
BASIS={tuple([1]+[0]*9)}
for i in range(1,10):
    r=[0]*10;r[i]=1;BASIS.add(tuple(r))
OFF=F.bm(F.PTS[:,0]!=0)

def escore(R):
    m=F.emask(R)
    return (int((m&OFF).bit_count()),int(m.bit_count()),F.score(m)[0])
def key(R):return tuple(sorted(tuple(map(int,r)) for r in R))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raycap',type=int,default=18);ap.add_argument('--seconds',type=int,default=1800);ap.add_argument('--seen-cap',type=int,default=100000);ap.add_argument('--out',required=True);a=ap.parse_args()
    C=F.smooth(ROOT);assert C is not None
    rs=escore(ROOT);assert rs[:2]==(20,8973),rs
    print('ROOT',rs,'rays',len(ROOT),'cones',len(C),flush=True)
    seen={key(ROOT)};heap=[];cnt=0;processed=0;t=time.time();best=(rs,ROOT,C,[])
    def push(R,C,h):
        nonlocal cnt
        s=escore(R);cnt+=1
        # prioritize true off-fiber count, then mod-2 defect, then total E
        heapq.heappush(heap,(s[0],s[2],s[1],len(R),cnt,R,C,h))
    push(ROOT,C,[])
    while heap and time.time()-t<a.seconds and len(seen)<a.seen_cap:
        off,pdef,e,n,_,R,C,h=heapq.heappop(heap);processed+=1
        s=(off,e,pdef)
        if (off,pdef,e)<(best[0][0],best[0][2],best[0][1]):
            best=(s,R,C,h);print('BEST',s,'rays',n,'processed',processed,'seen',len(seen),flush=True)
            if off==0:break
        # Exact contractions, but retain the fixed unimodular basis rays.
        for di in range(n):
            if tuple(map(int,R[di])) in BASIS:continue
            T=np.delete(R,di,axis=0);G=F.smooth(T)
            if G is None:continue
            k=key(T)
            if k in seen:continue
            seen.add(k);push(T,G,h+[{'op':'del','ray':list(map(int,R[di]))}])
        if n<a.raycap:
            for w,G,S in F.stars(R,C):
                T=np.vstack([R,w]);k=key(T)
                if k in seen:continue
                seen.add(k);push(T,G,h+[{'op':'add','face':[list(map(int,R[i])) for i in S],'w':list(map(int,w))}])
        if processed%50==0:
            print('PROG',processed,'seen',len(seen),'queue',len(heap),'top',heap[0][:4] if heap else None,'sec',round(time.time()-t,1),flush=True)
    out={'root_score':rs,'score':best[0],'rays':best[1].tolist(),'cones':[list(c) for c in best[2]],'hist':best[3],'processed':processed,'seen':len(seen),'queue':len(heap),'elapsed':time.time()-t,'raycap':a.raycap,'seen_cap':a.seen_cap}
    with open(a.out,'w') as f:json.dump(out,f,separators=(',',':'))
    print('FINAL',json.dumps({k:v for k,v in out.items() if k not in ('rays','cones','hist')}),flush=True)

if __name__=='__main__':main()
