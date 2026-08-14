import argparse,heapq,json,time
import numpy as np
import fcomp10 as F

ROOT=np.array([
[0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],
[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],
[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1],
[1,-1,0,0,0,0,0,0,0,0],[1,0,-1,0,0,0,0,0,0,0],[1,0,0,-1,0,0,0,0,0,0],
[1,0,0,0,-1,0,0,0,0,0],[1,0,0,0,0,-1,0,0,0,0],[1,0,0,0,0,0,-1,0,0,0],
[3,0,0,0,0,0,0,-1,-1,-1],[1,0,0,0,0,0,0,0,0,0],[-1,0,0,0,0,0,0,0,0,0]],dtype=int)
OFF=F.bm(F.PTS[:,0]!=0)

def sc(m): return ((m&OFF).bit_count(),m.bit_count())
def key(R): return tuple(sorted(tuple(map(int,r)) for r in R))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--raycap',type=int,default=23);ap.add_argument('--seconds',type=int,default=1500);ap.add_argument('--seen-cap',type=int,default=40000);ap.add_argument('--out',required=True);a=ap.parse_args()
 C=F.smooth(ROOT);assert C is not None
 m=F.emask(ROOT);root_sc=sc(m);print('ROOT',root_sc,'rays',len(ROOT),'cones',len(C),flush=True)
 heap=[];seen={key(ROOT)};cnt=0;processed=0;t=time.time();best=(root_sc,ROOT,C,[])
 def push(R,C,h):
  nonlocal cnt
  s=sc(F.emask(R));cnt+=1;heapq.heappush(heap,(s[0],s[1],len(R),cnt,R,C,h))
 push(ROOT,C,[])
 while heap and time.time()-t<a.seconds and len(seen)<a.seen_cap:
  off,e,n,_,R,C,h=heapq.heappop(heap);processed+=1
  if (off,e)<best[0]:
   best=((off,e),R,C,h);print('BEST',best[0],'rays',n,'processed',processed,'seen',len(seen),flush=True)
   if off==0:break
  if n>=a.raycap:continue
  children=F.stars(R,C)
  if processed<=10 or processed%20==0:print('EXPAND',processed,'score',(off,e),'rays',n,'stars',len(children),'sec',round(time.time()-t,1),flush=True)
  for w,G,S in children:
   T=np.vstack([R,w]);k=key(T)
   if k in seen:continue
   seen.add(k);push(T,G,h+[{'face':list(map(int,S)),'w':list(map(int,w))}])
  if processed%20==0:print('PROG',processed,len(seen),len(heap),'top',heap[0][:3] if heap else None,flush=True)
 out={'root_score':root_sc,'score':best[0],'rays':best[1].tolist(),'cones':[list(c) for c in best[2]],'hist':best[3],'processed':processed,'seen':len(seen),'queue':len(heap),'elapsed':time.time()-t,'raycap':a.raycap}
 with open(a.out,'w') as f:json.dump(out,f,separators=(',',':'))
 print('FINAL',json.dumps({k:v for k,v in out.items() if k not in ('rays','cones','hist')}),flush=True)

if __name__=='__main__':main()
