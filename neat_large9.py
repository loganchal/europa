import argparse,json,urllib.request,time
from neat_exact9 import polytopes,check_neat,D

URL='https://polymake.org/polytopes/paffenholz/data/smooth-fano/d9/fv-09-{}p.gz'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--facets',type=int,required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    path=f'/tmp/fv-09-{a.facets}p.gz';urllib.request.urlretrieve(URL.format(a.facets),path)
    selected=[];nonneat=[];unresolved=[];errors=[];scanned=0;t=time.time()
    for ordinal,U in enumerate(polytopes(path),1):
        scanned+=1
        widths=(-U[D:].sum(axis=1)).astype(int)
        box=1
        for w in widths: box*=2*int(w)+1
        if box<5000: continue
        res=check_neat(U,200000)
        rec={'ordinal':ordinal,'box':int(box),'rays':U.tolist(),**res}
        selected.append({'ordinal':ordinal,'box':int(box),'valid':res.get('valid'),'min_inter':res.get('min_inter'),'neat':res.get('neat'),'best_b':res.get('best_b')})
        if res.get('neat') is False:
            nonneat.append(rec);print('NONNEAT '+json.dumps(rec,separators=(',',':')),flush=True);break
        if res.get('unresolved_big'): unresolved.append(rec)
        elif 'error' in res: errors.append(rec)
        if len(selected)%10==0: print(json.dumps({'facets':a.facets,'large':len(selected),'last_box':box,'sec':round(time.time()-t,2)}),flush=True)
    out={'dimension':D,'facet_count':a.facets,'scanned':scanned,'large_count':len(selected),'selected':selected,'nonneat':nonneat,'unresolved':unresolved,'errors':errors,'elapsed':time.time()-t}
    with open(a.out,'w') as f: json.dump(out,f,separators=(',',':'))
    print('FINAL '+json.dumps({'facets':a.facets,'scanned':scanned,'large':len(selected),'nonneat':len(nonneat),'unresolved':len(unresolved),'errors':len(errors),'min_inter':min([x['min_inter'] for x in selected if x.get('min_inter') is not None] or [None],key=lambda z:10**9 if z is None else z),'elapsed':out['elapsed']},separators=(',',':')),flush=True)
if __name__=='__main__':main()
