#!/usr/bin/env python3
"""Run the documented experiments. python code/run_experiments.py [--quick]

Full run: fresh-reference R=1200 per audit condition, b=999; three seeds.
Power/robustness: independent bank per seed, conditionally independent test
observations. Their Clopper-Pearson intervals are CONDITIONAL on that bank.
"""
from pathlib import Path
import argparse, csv, json, platform, time
import numpy as np
import scipy
from scipy.stats import beta
from scipy.linalg import hadamard
from mask_profiles import *

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT/"results"
SEEDS = [20261018, 20261019, 20261020]
PERM_SEEDS = [20261021, 20261022, 20261023]


def write_csv(name, rows):
    if not rows: return
    with (RES/name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def ci(k, R):
    return (0. if k == 0 else beta.ppf(.025, k, R-k+1),
            1. if k == R else beta.ppf(.975, k+1, R-k))


def record(rows, meta, pv, scope, skip_q=False, only=None):
    for j, name in enumerate(METHODS):
        if only is not None and name not in only:
            continue
        if skip_q and name == "Q_upper":
            continue
        k = int(np.sum(pv[:, j]<=.05));R=len(pv);lo,hi=ci(k,R)
        rows.append(dict(**meta, method=name, R=R, rejected=k, rate=k/R,
                         ci_low=lo, ci_high=hi, ci_scope=scope))


def algebra_checks():
    rng = np.random.default_rng(530)
    maxima = dict(walsh_haar=0., trace_gram=0., centered_gram=0., commutator=0.,
                  product=0., mean_mc_error=0., covariance_mc_error=0.)
    for N in [2,4,8]:
        n=N*N;m=int(np.log2(N));w=hadamard(N)/np.sqrt(N);U=np.kron(w,w)
        s=np.count_nonzero(np.diff(w,axis=1),axis=1)
        q=np.maximum(s[:,None],s[None,:]).ravel()
        bands=[q==0]+[(q>=2**(k-1))&(q<2**k) for k in range(1,m+1)]
        D=[U[a].T@U[a] for a in bands]
        for _ in range(5):
            x=rng.normal(size=(N,N));mask=(rng.random((N,N))<.47).astype(float)
            if mask.sum()<2: mask.flat[:2]=1
            mu=mask.ravel();T=np.diag(mu);L=mu.sum()
            a=(w@x@w.T).ravel()**2
            maxima["walsh_haar"]=max(maxima["walsh_haar"],np.max(np.abs(
                np.array([a[b].sum() for b in bands])-energies(x))))
            h=np.array([[np.trace(di@T@dj@T) for dj in D] for di in D])
            maxima["trace_gram"]=max(maxima["trace_gram"],np.max(abs(h-mask_gram(mask))))
            C=T-np.outer(mu,mu)/L
            aa=np.array([np.trace(di@C) for di in D]);ell=L-1
            hh=np.array([[np.trace(di@C@dj@C) for dj in D] for di in D])
            dense=2/(ell*(ell+2))*(hh-np.outer(aa,aa)/ell)
            mean,cov=exact_moments(mask,True)
            maxima["centered_gram"]=max(maxima["centered_gram"],np.max(abs(cov-dense)))
            for r in range(m+1):
                P=sum(D[:r+1]);g=2**r;ss=N//g
                theta=mask.reshape(g,ss,g,ss).mean(axis=(1,3))
                value=np.sqrt(theta*(1-theta)).max()
                maxima["commutator"]=max(maxima["commutator"],abs(
                    np.linalg.norm(P@T-T@P,2)-value))
            dx=details(x);dm=details(mask)
            for k in range(1,m+1):
                rhs=projection(mask,k-1)*dx[k]+dm[k]*projection(x,k-1)
                def dk(z): return projection(z,k)-projection(z,k-1)
                rhs+=dk(dm[k]*dx[k])
                for j in range(k+1,m+1):rhs+=dk(dm[j]*dx[j])
                maxima["product"]=max(maxima["product"],np.max(abs(rhs-dk(mask*x))))
    N=8;mask=make_masks(N)["circle"]
    for centered in [False,True]:
        p=profiles(preprocess(rng.normal(size=(120000,N,N)),mask,centered))
        mean,cov=exact_moments(mask,centered)
        maxima["mean_mc_error"]=max(maxima["mean_mc_error"],np.max(abs(p.mean(0)-mean)))
        maxima["covariance_mc_error"]=max(maxima["covariance_mc_error"],np.max(abs(np.cov(p.T)-cov)))
    (RES/"algebra_checks.json").write_text(json.dumps(maxima,indent=2))
    assert max(maxima[k] for k in ["walsh_haar","trace_gram","centered_gram","commutator","product"])<1e-10
    print("algebra checks",maxima,flush=True)


def fresh_audit(b, Rseed):
    rows=[]
    conditions=[(N,"circle",True) for N in [16,32,64]]
    conditions += [(16,name,False) for name in ["full","circle","irregular","half","checker","random","one_pixel"]]
    for N,name,centered in conditions:
        mask=make_masks(N)[name]
        for seed in SEEDS:
            rng=np.random.default_rng(np.random.SeedSequence([seed,N,int(centered),list(make_masks(N)).index(name)]))
            matched=[];unmatched=[]
            for r in range(Rseed):
                z=rng.normal(size=(b+1,N,N))
                y=preprocess(z,mask,centered)
                st=statistics(y,mask,centered)
                matched.append(pvalues(st[:1],st[1:])[0])
                full=statistics(z[1:],np.ones_like(mask),False)
                unmatched.append(pvalues(st[:1],full)[0])
            meta=dict(N=N,mask=name,centered=int(centered),seed=seed,b=b)
            record(rows,dict(**meta,reference="matched"),np.array(matched),"unconditional_fresh")
            record(rows,dict(**meta,reference="unmatched"),np.array(unmatched),"unconditional_fresh",
                   only=["H_upper","H_two"])
            np.savez_compressed(RES/f"audit_raw_{N}_{name}_{int(centered)}_{seed}.npz",
                                matched=np.array(matched),unmatched_H=np.array(unmatched)[:,:2])
        write_csv("fresh_audit.csv",rows)
        print("fresh audit completed",N,name,centered,flush=True)


def shared_power(b,R):
    rows=[]
    rho_values=[0.,.05,.1,.25,.5,1.,2.,4.,8.,32.]
    conditions=[(N,"circle",True) for N in [16,32,64]]+[(16,"full",False)]
    for N,name,centered in conditions:
        mask=make_masks(N)[name];ell=int(mask.sum())-int(centered)
        for seed in SEEDS:
            rng=np.random.default_rng(np.random.SeedSequence([seed,N,int(centered),901]))
            ref=statistics(preprocess(rng.normal(size=(b,N,N)),mask,centered),mask,centered)
            raw={"reference_statistics":ref}
            for pattern,x in signals(N).items():
                if name=="full" and pattern!="spike":continue
                clean=preprocess(x,mask,centered);clean/=np.linalg.norm(clean)
                for rho in rho_values:
                    y=preprocess(rng.normal(size=(R,N,N)),mask,centered)+np.sqrt(rho*ell)*clean
                    st=statistics(y,mask,centered)
                    pv=pvalues(st,ref)
                    raw[f"{pattern}_rho_{rho:g}"]=pv
                    record(rows,dict(N=N,mask=name,centered=int(centered),seed=seed,b=b,
                                     pattern=pattern,rho=rho,Lambda=rho*ell),pv,"conditional_shared_bank")
            np.savez_compressed(RES/f"power_raw_{N}_{name}_{int(centered)}_{seed}.npz",**raw)
        write_csv("power.csv",rows)
        print("power completed",N,name,flush=True)


def robustness(b,R):
    rows=[];N=32;mask=make_masks(N)["circle"]
    for seed in SEEDS:
        rng=np.random.default_rng(np.random.SeedSequence([seed,830]))
        for model in ["gaussian","spherical_t5","iid_t5","iid_laplace","hetero","smooth","interpolate"]:
            z=rng.normal(size=(R,N,N))
            kind=model if model in ["hetero","smooth","interpolate"] else "identity"
            if model=="spherical_t5": z/=np.sqrt(rng.chisquare(5,size=(R,1,1))/5)
            elif model=="iid_t5":z=rng.standard_t(5,size=(R,N,N))*np.sqrt(3/5)
            elif model=="iid_laplace":z=rng.laplace(size=(R,N,N))/np.sqrt(2)
            y=preprocess(z,mask,True,kind)
            obs=statistics(y,mask,True,with_q=False)
            for refkind in ([kind,"identity"] if kind!="identity" else ["identity"]):
                ref=statistics(preprocess(rng.normal(size=(b,N,N)),mask,True,refkind),mask,True,with_q=False)
                pv=pvalues(obs,ref)
                np.savez_compressed(RES/f"robustness_raw_{model}_{refkind}_{seed}.npz",
                                    pvalues=pv[:,[0,1,3,4]],reference_statistics=ref[:,[0,2]])
                record(rows,dict(N=N,seed=seed,b=b,model=model,reference_map=refkind),pv,"conditional_shared_bank",skip_q=True)
    write_csv("robustness.csv",rows)
    print("robustness completed",flush=True)


def permutation_comparison(b,R,seeds=None,tag=""):
    # Uniform retained-pixel permutations. Valid for iid noise even if not spherical.
    rows=[];N=16;mask=make_masks(N)["circle"];ix=np.flatnonzero(mask.ravel())
    for seed in (PERM_SEEDS if seeds is None else seeds):
        rng=np.random.default_rng(np.random.SeedSequence([seed,345]))
        for model in ["gaussian","iid_t5","iid_laplace"]:
            pvg=[];pvp=[]
            for i in range(R):
                z=rng.normal(size=(N,N))
                if model=="iid_t5":z=rng.standard_t(5,size=(N,N))*np.sqrt(3/5)
                elif model=="iid_laplace":z=rng.laplace(size=(N,N))/np.sqrt(2)
                y=preprocess(z,mask,True)
                obs=statistics(y[None],mask,True,False)
                ref=statistics(preprocess(rng.normal(size=(b,N,N)),mask,True),mask,True,False)
                pvg.append(pvalues(obs,ref)[0])
                vals=y.ravel()[ix]
                arr=np.zeros((b,N*N))
                for j in range(b):arr[j,ix]=rng.permutation(vals)
                ref=statistics(arr.reshape(b,N,N),mask,True,False)
                pvp.append(pvalues(obs,ref)[0])
            meta=dict(N=N,seed=seed,b=b,model=model)
            record(rows,dict(**meta,reference="Gaussian_MC"),np.array(pvg),"unconditional_fresh",skip_q=True)
            record(rows,dict(**meta,reference="retained_permutation"),np.array(pvp),"unconditional_fresh",skip_q=True)
            np.savez_compressed(RES/f"{tag}permutation_raw_{model}_{seed}.npz",
                                Gaussian_MC=np.array(pvg)[:,[0,1,3,4]],
                                retained_permutation=np.array(pvp)[:,[0,1,3,4]])
        write_csv(tag+"permutation.csv",rows)
    print("permutation comparison completed",flush=True)


def timings():
    rows=[];rng=np.random.default_rng(420)
    for N in [16,32,64,128,256]:
        mask=make_masks(N)["circle"];x=rng.normal(size=(N,N))
        for op,fn in [("profile",lambda:profiles(x)),("uncentered_moments",lambda:exact_moments(mask)),
                      ("centered_moments",lambda:exact_moments_direct(mask,True))]:
            samples=[]
            for _ in range(7):
                t=time.perf_counter()
                for j in range(15):fn()
                samples.append((time.perf_counter()-t)/15)
            rows.append(dict(N=N,operation=op,median_ms=1000*np.median(samples),
                             min_ms=1000*min(samples),max_ms=1000*max(samples)))
    write_csv("timings.csv",rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--quick",action="store_true")
    ap.add_argument("--stage",choices=["all","checks","audit","power","robustness","permutation","permutation_pilot","timings"],default="all")
    args=ap.parse_args();RES.mkdir(exist_ok=True)
    b=99 if args.quick else 999
    t=time.perf_counter()
    actions={"checks":lambda:algebra_checks(),"audit":lambda:fresh_audit(b,12 if args.quick else 400),
             "power":lambda:shared_power(b,80 if args.quick else 1000),
             "robustness":lambda:robustness(b,100 if args.quick else 3000),
             "permutation":lambda:permutation_comparison(b,10 if args.quick else 1000),
             "timings":timings}
    for name,fn in actions.items():
        if args.stage in ["all",name]:fn()
    if args.stage == "permutation_pilot":
        permutation_comparison(b,200,seeds=SEEDS,tag="pilot_repeat_")
    manifest=dict(seeds=SEEDS,permutation_seeds=PERM_SEEDS,b=b,quick=args.quick,stage=args.stage,
                  elapsed_seconds=time.perf_counter()-t,python=platform.python_version(),
                  numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform())
    (RES/("manifest_"+args.stage+".json")).write_text(json.dumps(manifest,indent=2))
    print("DONE",manifest,flush=True)


if __name__=="__main__":main()
