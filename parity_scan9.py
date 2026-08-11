import argparse,gzip,itertools,json,time,urllib.request
import numpy as np
D=9
PTS=np.array(list(itertools.product((-1,0,1),repeat=D)),dtype=np.int16)
FULL=(1<<len(PTS))-1
URL='https://polymake.org/polytopes/paffenholz/data/smooth-fano/d9/fv-09-{}p.gz'

def bm(ok):return int.from_bytes(np.packbits(ok,bitorder='little').tobytes(),'little')&FULL
MOD=PTS%2
HM=[]
for code in range(1,1<<D):
 v=np.array([(code>>i)&1 for i in range(D)],dtype=np.int16)
 HM.append((code,bm((MOD@v)%2==1)))
RC={}
def rmask(r):
 k=tuple(map(int,r));q=RC.get(k)
 if q is None:
  q=bm(np.abs(PTS@np.array(k,dtype=np.int16))<=1);RC[k]=q
 return q

def polytopes(path):
 cur=[];inside=False
 with gzip.open(path,'rt') as f:
  for raw in f:
   s=raw.strip()
   if not inside:
    if not s:continue
    if s!='FACETS':raise ValueError(s)
    inside=True;cur=[];continue
   if not s:
    yield cur;inside=False;continue
   a=[int(x) for x in s.split()]
   if len(a)!=D+1 or a[0]!=1:raise ValueError(s)
   cur.append(tuple(-x for x in a[1:]))
 if inside:yield cur

def score(rays):
 m=FULL
 for r in rays:m&=rmask(r)
 best=(10**9,None)
 for code,h in HM:
  c=(m&h).bit_count()
  if c<best[0]:best=(c,code)
 return best[0],m.bit_count(),best[1]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--facets',type=int,required=True);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);ap.add_argument('--out',required=True);a=ap.parse_args()
 path=f'/tmp/fv-09-{a.facets}p.gz';urllib.request.urlretrieve(URL.format(a.facets),path)
 best=[];selected=0;total=0;t=time.time()
 for ordinal,rays in enumerate(polytopes(path),1):
  total+=1
  if (ordinal-1)%a.shards!=a.shard:continue
  selected+=1;s=score(rays);best.append((s[0],s[1],ordinal,s[2],rays));best.sort(key=lambda x:(x[0],x[1]));best=best[:20]
  if selected%25000==0:print(json.dumps({'f':a.facets,'shard':a.shard,'selected':selected,'best':best[0][:4],'sec':round(time.time()-t,1)}),flush=True)
 out={'facets':a.facets,'shard':a.shard,'shards':a.shards,'total':total,'selected':selected,'best':[{'parity':x[0],'E':x[1],'ordinal':x[2],'code':x[3],'rays':[list(r) for r in x[4]]} for x in best],'elapsed':time.time()-t}
 with open(a.out,'w') as f:json.dump(out,f,separators=(',',':'))
 print('FINAL '+json.dumps({'facets':a.facets,'shard':a.shard,'selected':selected,'best':best[0][:4],'elapsed':out['elapsed']}),flush=True)
if __name__=='__main__':main()
