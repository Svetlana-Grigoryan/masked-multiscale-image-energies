"""Reconstruct every new p-value, rejection count, and conditional interval."""
from pathlib import Path
import csv,json
import numpy as np
from scipy.stats import beta
from mask_profiles import pvalues
ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results'

def main():
    rows=list(csv.DictReader((RES/'review_paired_experiments.csv').open()));cache={};err=0.
    for row in rows:
        seed=row['seed'];rho=f"{float(row['rho']):g}";sig=row['signal']
        name=(f'paired_reference_raw_{seed}_{sig}_{rho}.npz' if row['study']=='paired' else f'patch_raw_{seed}_{rho}.npz')
        if name not in cache:
            with np.load(RES/name) as a:
                if row['study']=='paired':
                    cache[name]={}
                    for tag,key in [('u','geometric'),('mean','adjusted')]:
                        pv=pvalues(a[key+'_statistics'],a[key+'_references'])
                        err=max(err,float(np.max(abs(pv-a[key+'_pvalues']))));cache[name][tag]=pv
                else:
                    pv=pvalues(a['statistics'],a['references']);err=max(err,float(np.max(abs(pv-a['pvalues']))));cache[name]={'u=mean':pv}
        j={'upper':0,'two':1,'peak':3,'combined':4}[row['method']]
        v=cache[name][row['reference']][:,j];k=int(np.count_nonzero(v<=.05));n=len(v)
        assert k==int(row['rejections']) and n==int(row['R'])
        assert abs(k/n-float(row['rate']))<1e-14
        lo=0. if k==0 else beta.ppf(.025,k,n-k+1);hi=1. if k==n else beta.ppf(.975,k+1,n-k)
        assert abs(lo-float(row['lo']))<1e-12 and abs(hi-float(row['hi']))<1e-12
    assert err==0
    result=dict(new_result_rows_verified=len(rows),new_raw_files_verified=len(cache),maximum_rank_regeneration_error=err,
                uncertainty='Intervals conditional on each shared bank; identical rho-zero signal rows are not independent.')
    (RES/'review_result_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
