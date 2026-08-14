import json,os,re,requests,time

REPO=os.environ['GITHUB_REPOSITORY']
TOKEN=os.environ['GITHUB_TOKEN']
RUN=31478691714
H={'Authorization':f'Bearer {TOKEN}','Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'}
base='https://api.github.com'

jobs=[]
for page in (1,2,3):
    r=requests.get(f'{base}/repos/{REPO}/actions/runs/{RUN}/jobs',headers=H,params={'per_page':100,'page':page},timeout=60)
    r.raise_for_status(); part=r.json()['jobs']; jobs.extend(part)
    if len(part)<100: break
print('JOBS',len(jobs),flush=True)
assert len(jobs)==182, len(jobs)

rows=[]; missing=[]
for k,j in enumerate(jobs,1):
    jid=j['id']; name=j['name']
    r=requests.get(f'{base}/repos/{REPO}/actions/jobs/{jid}/logs',headers=H,timeout=90,allow_redirects=True)
    r.raise_for_status()
    text=r.content.decode('utf-8-sig','replace')
    finals=[]
    for line in text.splitlines():
        p=line.find('FINAL ')
        if p>=0:
            s=line[p+6:].strip()
            try: finals.append(json.loads(s))
            except Exception: pass
    if not finals:
        missing.append({'id':jid,'name':name,'bytes':len(r.content),'head':text[-500:]})
        continue
    rec=finals[-1]; rec['job_id']=jid; rec['job_name']=name; rows.append(rec)
    if k%20==0: print('PROG',k,'rows',len(rows),'missing',len(missing),flush=True)

out={'run':RUN,'jobs':len(jobs),'parsed':len(rows),'missing':missing,
     'selected_total':sum(int(x.get('selected',0)) for x in rows),
     'nonneat_total':sum(int(x.get('nonneat',0)) for x in rows),
     'unresolved_total':sum(int(x.get('unresolved',0)) for x in rows),
     'errors_total':sum(int(x.get('errors',0)) for x in rows),
     'rows':rows}
finite=[x for x in rows if isinstance(x.get('best_inter'),int) and x['best_inter']<10**9]
out['global_best']=min(finite,key=lambda x:x['best_inter']) if finite else None
byfacet={}
for x in rows:
    f=str(x['facets']); y=byfacet.setdefault(f,{'selected':0,'nonneat':0,'unresolved':0,'errors':0,'best_inter':10**9,'best_row':None})
    y['selected']+=int(x.get('selected',0));y['nonneat']+=int(x.get('nonneat',0));y['unresolved']+=int(x.get('unresolved',0));y['errors']+=int(x.get('errors',0))
    b=x.get('best_inter')
    if isinstance(b,int) and b<y['best_inter']: y['best_inter']=b;y['best_row']=x
out['by_facet']=byfacet
open('old-neat-log-audit.json','w').write(json.dumps(out,separators=(',',':')))
print('FINAL_AUDIT',json.dumps({k:v for k,v in out.items() if k not in ('rows','missing','by_facet')}),flush=True)
print('GLOBAL_BEST',json.dumps(out['global_best']),flush=True)
print('BY_FACET',json.dumps({k:{a:b for a,b in v.items() if a!='best_row'} for k,v in byfacet.items()},sort_keys=True),flush=True)
assert not missing, missing[:3]
assert out['nonneat_total']==0 and out['unresolved_total']==0 and out['errors_total']==0, out
