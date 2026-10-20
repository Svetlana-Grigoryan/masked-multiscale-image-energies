"""Verify delivered count tables against saved p-value arrays and CI endpoints."""
from pathlib import Path
import csv,json
import numpy as np
from scipy.stats import beta

ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results'
METHODS=['H_upper','H_two','Q_upper','peak','combined']
SHORT=['H_upper','H_two','peak','combined']


def main():
    checks={}
    for stage in ['fresh_audit','power','robustness','permutation']:
        with (RES/(stage+'.csv')).open() as f:rows=list(csv.DictReader(f))
        total=0
        for r in rows:
            if stage=='fresh_audit':
                path=RES/f"audit_raw_{r['N']}_{r['mask']}_{r['centered']}_{r['seed']}.npz"
                a=np.load(path)
                if r['reference']=='matched':v=a['matched'][:,METHODS.index(r['method'])]
                else:v=a['unmatched_H'][:,['H_upper','H_two'].index(r['method'])]
            elif stage=='power':
                path=RES/f"power_raw_{r['N']}_{r['mask']}_{r['centered']}_{r['seed']}.npz"
                a=np.load(path);key=f"{r['pattern']}_rho_{float(r['rho']):g}"
                v=a[key][:,METHODS.index(r['method'])]
            elif stage=='robustness':
                path=RES/f"robustness_raw_{r['model']}_{r['reference_map']}_{r['seed']}.npz"
                a=np.load(path);v=a['pvalues'][:,SHORT.index(r['method'])]
            else:
                path=RES/f"permutation_raw_{r['model']}_{r['seed']}.npz"
                a=np.load(path);v=a[r['reference']][:,SHORT.index(r['method'])]
            assert np.all(np.isfinite(v)) and np.all((v>=0)&(v<=1)),r
            k=int(np.count_nonzero(v<=.05));R=len(v)
            assert k==int(r['rejected']) and R==int(r['R']),r
            assert abs(k/R-float(r['rate']))<1e-13,r
            lo=0 if k==0 else beta.ppf(.025,k,R-k+1)
            hi=1 if k==R else beta.ppf(.975,k+1,R-k)
            assert abs(lo-float(r['ci_low']))<1e-12,r
            assert abs(hi-float(r['ci_high']))<1e-12,r
            a.close();total+=1
        checks[stage+'_rows_verified']=total
    for name in ['main.log']:
        if not (ROOT/name).exists():continue
        txt=(ROOT/name).read_text()
        for term in ['Overfull','undefined','Fatal']:
            assert term not in txt,(name,term)
    checks['compilation_logs_clean']=bool((ROOT/'main.log').exists())
    checks['note']='Counts verified from raw p-values; not a certification of novelty or proof correctness.'
    (RES/'delivery_validation.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
