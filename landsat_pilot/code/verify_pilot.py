"""Check all saved counts/intervals, all power ranks, selected fresh ranks."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
from scipy.stats import beta
from run_pilot import ROOT,DATA,RES,NAMES,METHODS,B,scaled_mask,compute_stats,ci
from mask_profiles import pvalues,preprocess,exact_moments

def main():
    meta=json.loads((RES/'protocol_and_provenance.json').read_text())
    assert hashlib.sha256((DATA/'RGB.byte.tif').read_bytes()).hexdigest()==meta['source_sha256']
    assert 'CC0 1.0' in (DATA/'RASTERIO_DATA_README.rst').read_text()
    checked={};maxerr=0.;cache={}
    for stage in ['fresh_audit','power']:
        rows=list(csv.DictReader((RES/(stage+'.csv')).open()))
        for r in rows:
            path=(RES/(f"audit_raw_{r['mask']}_{r['seed']}.npz" if stage=='fresh_audit' else f"power_raw_{r['mask']}_{r['seed']}.npz"))
            if path not in cache:
                a=dict(np.load(path));cache[path]=a
                if stage=='power':
                    for key in list(a):
                        if key.startswith('pvalues_'):
                            expected=pvalues(a['statistics_'+key[8:]],a['references'])
                            maxerr=max(maxerr,float(np.max(abs(a[key]-expected))))
            a=cache[path]
            vals=(a[r['reference']] if stage=='fresh_audit' else a[f"pvalues_{float(r['rho']):g}"])[:,METHODS.index(r['method'])]
            assert np.all((vals>=0)&(vals<=1))
            k=int(np.count_nonzero(vals<=.05));n=len(vals);lo,hi=ci(k,n)
            assert k==int(r['rejected']) and n==int(r['R'])
            assert abs(k/n-float(r['rate']))<1e-14
            assert abs(lo-float(r['ci_low']))<1e-12 and abs(hi-float(r['ci_high']))<1e-12
        checked[stage+'_rows']=len(rows)
    assert maxerr==0
    # Independently regenerate the first three fresh tests of every seed/mask.
    nreg=0
    for idx,name in enumerate(NAMES):
        mask=scaled_mask(name,64);full=np.ones_like(mask);mu,cov=exact_moments(mask,True)
        for seed in meta['seeds']:
            rng=np.random.default_rng(np.random.SeedSequence([seed,64,idx,191]))
            a=np.load(RES/f'audit_raw_{name}_{seed}.npz')
            for rep in range(3):
                z=rng.normal(size=(B+1,64,64));obs=compute_stats(preprocess(z[:1],mask,True),mask,mu,cov)
                for label,refmask in [('matched',mask),('unmatched',full)]:
                    refs=compute_stats(preprocess(z[1:],refmask,True),mask,mu,cov)
                    assert np.array_equal(pvalues(obs,refs)[0],a[label][rep])
                nreg+=1
    benchmark=list(csv.DictReader((RES/'benchmark.csv').open()))
    assert len(benchmark)==20
    for r in benchmark:
        assert int(r['L'])==int(scaled_mask(r['mask'],int(r['N'])).sum())
        assert float(r['min_ms'])<=float(r['median_ms'])<=float(r['max_ms'])
        assert float(r['peak_RSS_MiB'])>0
    agreements=list(csv.DictReader((RES/'numerical_agreement.csv').open()))
    assert len(agreements)==15
    for r in agreements:
        assert float(r['covariance_max_abs'])<1e-12
    result=dict(**checked,all_power_rank_max_error=maxerr,fresh_tests_regenerated=nreg,
                numerical_pairs=len(agreements),benchmark_rows=len(benchmark),source_hash_verified=True,
                note='Original stochastic studies not rerun. This separate pilot checks simulated added noise, not original Landsat instrument noise.')
    (RES/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
