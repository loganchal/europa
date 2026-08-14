import argparse,json,os,tempfile,time,urllib.request
import neat_filter9 as nf


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--facets',type=int,required=True)
    ap.add_argument('--start',type=int,required=True)
    ap.add_argument('--end',type=int,required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--exhaust-cap',type=int,default=5000)
    ap.add_argument('--beam',type=int,default=64)
    a=ap.parse_args()
    url=nf.URL.format(a.facets)
    tmp=tempfile.NamedTemporaryFile(suffix='.gz',delete=False);tmp.close()
    urllib.request.urlretrieve(url,tmp.name)
    t=time.time(); scanned=nonrigid=candidate_polys=verified_false=large=0
    errors=[]; witness=None; maxbox=1; first_seen=last_seen=None
    for idx,U in enumerate(nf.records(tmp.name),start=1):
        if idx<a.start: continue
        if idx>a.end: break
        if first_seen is None:first_seen=idx
        last_seen=idx;scanned+=1
        w=nf.widths_for(U)
        if w is None:
            errors.append({'index':idx,'error':'not_standard'});continue
        cands,box,mode=nf.candidate_assignments(U,w,a.exhaust_cap,a.beam)
        maxbox=max(maxbox,int(box))
        if box>1:nonrigid+=1
        if box>a.exhaust_cap:large+=1
        if cands:
            candidate_polys+=1
            for qs in cands:
                ok,detail=nf.exact_fan_preserved(U,qs)
                if ok:
                    witness={'facet_count':a.facets,'index_in_file':idx,'normals':U.tolist(),'b':[0]*nf.D+list(qs),'widths':w.tolist(),'box':int(box),'verification':detail}
                    print('WITNESS '+json.dumps(witness,separators=(',',':')),flush=True)
                    break
                verified_false+=1
        if witness:break
        if scanned%100000==0:
            print(json.dumps({'facets':a.facets,'start':a.start,'end':a.end,'scanned':scanned,'index':idx,'candidates':candidate_polys,'large':large,'sec':round(time.time()-t,1)}),flush=True)
    os.unlink(tmp.name)
    expected=a.end-a.start+1
    out={'dimension':nf.D,'facet_count':a.facets,'start':a.start,'end':a.end,'expected':expected,'scanned':scanned,'first_seen':first_seen,'last_seen':last_seen,'nonrigid':nonrigid,'candidate_polys':candidate_polys,'verified_false_candidates':verified_false,'large_boxes':large,'witness':witness,'errors':errors[:20],'error_count':len(errors),'max_box':maxbox,'elapsed_sec':time.time()-t}
    with open(a.out,'w') as f:json.dump(out,f,separators=(',',':'))
    print('FINAL '+json.dumps({k:v for k,v in out.items() if k not in ('witness','errors')}|{'has_witness':witness is not None},separators=(',',':')),flush=True)
    if witness is None:
        assert scanned==expected and first_seen==a.start and last_seen==a.end, out
        assert not errors, out

if __name__=='__main__':main()
