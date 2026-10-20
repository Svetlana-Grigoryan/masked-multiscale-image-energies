"""Deterministic extension checks and measured direct/tree timings.

Creates only extension_checks.json, extension_summary.tex, tree_timings.csv,
and tree_timing_table.tex. It never changes the archived stochastic studies.
"""
from pathlib import Path
import argparse,csv,itertools,json,platform,time
import numpy as np
import scipy
from mask_profiles import (dimensions,details,energies,exact_moments,
 exact_moments_direct,mask_detail_gram_tree,make_masks,iid_energy_covariance,
 statistics,preprocess,pvalues,signals)
ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results'


def set_partitions(items):
    if not items:
        yield []
        return
    x=items[0]
    for rest in set_partitions(items[1:]):
        yield [[x]]+[list(b) for b in rest]
        for j in range(len(rest)):
            blocks=[list(b) for b in rest];blocks[j]=[x]+blocks[j]
            yield blocks


def avg_projector(blocks,n):
    P=np.zeros((n,n))
    for b in blocks:P[np.ix_(b,b)]=1/len(b)
    return P


def checks():
    rng=np.random.default_rng(20261019)
    out={'tree_masks_checked':0,'tree_gram_max_abs_error':0.,'centered_cov_max_abs_error':0.,
         'dense_centered_cov_max_abs_error':0.,'iid_cov_max_abs_error':0.,
         'hierarchies_checked':0,'regular_hierarchies':0,'nonregular_hierarchies':0}
    for N in [2,4,8,16,32,64]:
        masks=([np.array(v,dtype=float).reshape(2,2) for v in itertools.product([0,1],repeat=4)]
               if N==2 else [rng.integers(0,2,(N,N)).astype(float) for _ in range(64)])
        for mask in masks:
            if mask.sum()==0:continue
            out['tree_masks_checked']+=1
            D=details(mask).reshape(len(dimensions(N)),-1)
            g=(D*mask.ravel())@D.T
            out['tree_gram_max_abs_error']=max(out['tree_gram_max_abs_error'],float(np.max(abs(g-mask_detail_gram_tree(mask)))))
            if mask.sum()>1:
                mu,cov=exact_moments(mask,True);mud,covd=exact_moments_direct(mask,True)
                out['centered_cov_max_abs_error']=max(out['centered_cov_max_abs_error'],float(np.max(abs(cov-covd))),float(np.max(abs(mu-mud))))
                if N<=8:
                    n=N*N;P=[np.ones((n,n))/n]
                    coordinates=np.indices((N,N)).reshape(2,-1).T
                    for r in range(1,N.bit_length()):
                        labels=coordinates//(N//2**r)
                        P.append(np.all(labels[:,None,:]==labels[None,:,:],axis=2)/(n/4**r))
                    Ds=[P[0]]+[P[k]-P[k-1] for k in range(1,len(P))]
                    v=mask.ravel();L=v.sum();C=np.diag(v)-np.outer(v,v)/L;ell=L-1
                    aa=np.array([np.trace(d@C) for d in Ds]);HH=np.array([[np.trace(d@C@f@C) for f in Ds] for d in Ds])
                    dense=2/(ell*(ell+2))*(HH-np.outer(aa,aa)/ell)
                    out['dense_centered_cov_max_abs_error']=max(out['dense_centered_cov_max_abs_error'],float(np.max(abs(dense-cov))))
    # Exhaustive two-step strict hierarchies on six points, every binary mask.
    n=6;V=np.array(list(itertools.product([0.,1.],repeat=n)))
    for blocks in set_partitions(list(range(n))):
        if len(blocks)<2 or any(len(b)<2 for b in blocks):continue
        P0=np.ones((n,n))/n;P1=avg_projector(blocks,n);Ds=[P0,P1-P0,np.eye(n)-P1]
        regular=len(set(map(len,blocks)))==1
        scalar=True
        for k,j in [(0,1),(0,2),(1,2)]:
            H=np.einsum('bi,ij,bj->b',V,Ds[k]*Ds[j],V)
            e=np.einsum('bi,ij,bj->b',V,Ds[j],V)
            scalar &= bool(np.max(abs(H-np.trace(Ds[k])/n*e))<1e-12)
            if (k,j)==(1,2):
                local=np.zeros(len(V))
                for b in blocks:
                    dj=V@Ds[j];local+=(1/len(b)-1/n)*np.sum(dj[:,b]**2,axis=1)
                assert np.max(abs(H-local))<1e-12
        assert scalar==regular
        out['hierarchies_checked']+=1
        out['regular_hierarchies' if regular else 'nonregular_hierarchies']+=1
    # Exhaustive Rademacher fourth-cumulant checks, all nonempty 2x2 masks.
    N=2;Z=np.array(list(itertools.product([-1.,1.],repeat=4))).reshape(-1,2,2)
    for v in itertools.product([0.,1.],repeat=4):
        mask=np.array(v).reshape(2,2)
        if mask.sum()==0:continue
        for centered in [False,True]:
            if centered and mask.sum()<2:continue
            E=energies(preprocess(Z,mask,centered));emp=E.T@E/len(E)-np.outer(E.mean(0),E.mean(0))
            predicted=iid_energy_covariance(mask,-2,centered=centered)
            out['iid_cov_max_abs_error']=max(out['iid_cov_max_abs_error'],float(np.max(abs(predicted-emp))))
    # Confirm preserved first-seed centered fresh ranks under the new implementation.
    N=16;b=999;seed=20261018;mask=make_masks(N)['circle']
    rng=np.random.default_rng(np.random.SeedSequence([seed,N,1,list(make_masks(N)).index('circle')]))
    a=np.load(RES/'audit_raw_16_circle_1_20261018.npz');err=0.
    for r in range(12):
        z=rng.normal(size=(b+1,N,N));st=statistics(preprocess(z,mask,True),mask,True)
        err=max(err,float(np.max(abs(pvalues(st[:1],st[1:])[0]-a['matched'][r]))))
    out['regenerated_centered_fresh_trials']=12;out['regenerated_rank_max_abs_error']=err
    assert err==0
    assert out['tree_gram_max_abs_error']<1e-8
    assert out['centered_cov_max_abs_error']<1e-12
    assert out['dense_centered_cov_max_abs_error']<1e-12
    assert out['iid_cov_max_abs_error']<1e-12
    out.update(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform())
    (RES/'extension_checks.json').write_text(json.dumps(out,indent=2))
    def sci(x):
        if x==0:return '0'
        a,b=f'{x:.2e}'.split('e');return a+r'\times10^{'+str(int(b))+'}'
    text=(f"The added deterministic checks cover {out['tree_masks_checked']} nonempty square-grid masks, including all 15 nonempty masks at $N=2$, and 64 fixed-seed masks at each of $N=4,8,16,32,64$. "
          f"The largest absolute tree/direct Gram discrepancy is ${sci(out['tree_gram_max_abs_error'])}$; the centered normalized covariance discrepancy is ${sci(out['centered_cov_max_abs_error'])}$, and its dense-projector discrepancy is ${sci(out['dense_centered_cov_max_abs_error'])}$. "
          f"All {out['hierarchies_checked']} strictly splitting two-step hierarchies on six points without singleton intermediate blocks pass the characterization check over all 64 binary masks. "
          f"The exhaustive Rademacher covariance discrepancy is ${sci(out['iid_cov_max_abs_error'])}$. The first 12 centered fresh-reference ranks are reproduced exactly with the new algorithm.\n")
    (RES/'extension_summary.tex').write_text(text)
    print(json.dumps(out,indent=2),flush=True)


def timings():
    rows=[]
    for N in [16,32,64,128,256,512]:
        mask=make_masks(N)['circle'];row={'N':N}
        for name,fn in [('direct',exact_moments_direct),('tree',exact_moments)]:
            fn(mask,True);samples=[]
            for _ in range(7):
                start=time.perf_counter()
                for rep in range(3):fn(mask,True)
                samples.append((time.perf_counter()-start)/3)
            row[name+'_ms']=1000*float(np.median(samples))
        row['ratio_direct_tree']=row['direct_ms']/row['tree_ms'];rows.append(row)
    with (RES/'tree_timings.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    tex=r'''\begin{table}[htbp]\centering\small
\caption{Additional centered-moment timings in milliseconds: original direct method versus the tree algorithm, measured in the same environment. The final column is direct time divided by tree time.}\label{tab:treeTiming}
\begin{tabular}{rrrr}\toprule
$N$ & Direct & Tree & Ratio\\\midrule
'''
    tex+='\n'.join(f"{r['N']} & {r['direct_ms']:.3f} & {r['tree_ms']:.3f} & {r['ratio_direct_tree']:.2f}"+r'\\' for r in rows)
    tex+='\n'+r'\bottomrule\end{tabular}\end{table}'+'\n'
    (RES/'tree_timing_table.tex').write_text(tex)
    print('tree timings',rows,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checks-only',action='store_true');args=parser.parse_args()
    checks()
    if not args.checks_only:timings()
