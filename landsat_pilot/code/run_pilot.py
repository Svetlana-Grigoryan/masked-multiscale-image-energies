"""Real-scene mask pilot. Frozen CC0 Landsat fixture + explicitly added noise.

No claim is made about the original instrument noise, cloud segmentation,
reconstruction, or performance on multiple independent satellite scenes.
"""
from pathlib import Path
import argparse,csv,hashlib,json,os,platform,resource,subprocess,sys,time
import numpy as np
from PIL import Image
from scipy.stats import beta
from mask_profiles import exact_moments,exact_moments_direct,mask_detail_gram_tree,details,profiles,preprocess,statistics,pvalues,dimensions
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data';RES=ROOT/'results';FIG=ROOT/'figures'
SEEDS=[20261031,20261032,20261033];B=999;R_AUDIT=150;R_POWER=1000;LEVEL=.05
NAMES=['footprint','stripes','elliptical_gaps'];METHODS=['H_upper','H_two','Q_upper','peak','combined']
RHOS=[0.,.001,.005,.01,.025,.05,.1,.25,1.]

def savecsv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def ci(k,r):
    return (0. if k==0 else float(beta.ppf(.025,k,r-k+1)),1. if k==r else float(beta.ppf(.975,k+1,r-k)))

def row(stats,**meta):
    k=int(np.count_nonzero(stats<=LEVEL));r=len(stats);lo,hi=ci(k,r)
    return dict(**meta,rejected=k,R=r,rate=k/r,ci_low=lo,ci_high=hi)

def prepare():
    rgb=np.array(Image.open(DATA/'RGB.byte.tif'))
    source_mask=np.all(rgb!=0,axis=-1)
    # Fixed linear grayscale in stored digital-number units, no radiance claim.
    gray=np.einsum('ijk,k->ij',rgb.astype(np.float64),np.array([.2126,.7152,.0722]))/255
    n=1024;top=(n-rgb.shape[0])//2;left=(n-rgb.shape[1])//2
    image=np.zeros((n,n));foot=np.zeros((n,n),dtype=bool)
    image[top:top+rgb.shape[0],left:left+rgb.shape[1]]=gray
    foot[top:top+rgb.shape[0],left:left+rgb.shape[1]]=source_mask
    yy,xx=np.indices((n,n));t=(xx+.5)/n;v=(yy+.5)/n
    stripe=(xx%64)>=12
    excluded=(((t-.35)/.13)**2+((v-.38)/.11)**2<1)|(((t-.66)/.14)**2+((v-.64)/.10)**2<1)
    masks=dict(footprint=foot,stripes=foot&stripe,elliptical_gaps=foot&~excluded)
    np.savez_compressed(DATA/'fixed_scene.npz',rgb=rgb,gray=image,**{k:m.astype(np.uint8) for k,m in masks.items()})
    # Pixel-center decimation to 64^2. No interpolating map or fitted mask.
    ix=np.arange(64)*16+8
    np.savez_compressed(DATA/'fixed_scene_N64.npz',gray=image[np.ix_(ix,ix)],
                        **{k:m[np.ix_(ix,ix)].astype(np.uint8) for k,m in masks.items()})
    meta=dict(source='USGS Landsat 7 ETM-derived Rasterio RGB.byte.tif',
        source_url='https://raw.githubusercontent.com/rasterio/rasterio/1.4.3/tests/data/RGB.byte.tif',
        license_url='https://raw.githubusercontent.com/rasterio/rasterio/1.4.3/tests/data/README.rst',
        license='CC0 1.0 Universal Public Domain Dedication',source_shape=list(rgb.shape),
        source_sha256=hashlib.sha256((DATA/'RGB.byte.tif').read_bytes()).hexdigest(),
        crs='EPSG:32618',pixel_spacing_m=[300.0379266750948,300.041782729805],
        acquisition_date='not established from the supplied fixture',
        padding=dict(canvas=1024,top=top,left=left),
        source_validity_rule='all three stored RGB bands are nonzero; nodata is 0 in GeoTIFF metadata',
        cloud_detection='not performed; existing clouds remain part of the frozen template',
        grayscale_weights=[.2126,.7152,.0722],noise_injection='iid standard Gaussian after fixed pixel decimation; no model for original sensor noise',
        seeds=SEEDS,b=B,audit_trials_per_seed=R_AUDIT,power_trials_per_seed=R_POWER,rhos=RHOS,
        noise_design='fresh banks in level audit; one independent shared bank per seed and mask in power study',
        masks={k:dict(L=int(m.sum()),retained_fraction=float(m.mean()),L64=int(m[np.ix_(ix,ix)].sum())) for k,m in masks.items()},
        python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
        cpu_model=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),'unknown'),
        threads={k:os.environ.get(k,'unset') for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']})
    (RES/'protocol_and_provenance.json').write_text(json.dumps(meta,indent=2))
    print('prepared',json.dumps(meta['masks']),flush=True)

def scaled_mask(name,N):
    with np.load(DATA/'fixed_scene.npz') as a:base=a[name]
    if N<=1024:
        step=1024//N;ix=np.arange(N)*step+step//2;mask=base[np.ix_(ix,ix)]
    else:mask=np.tile(base,(N//1024,N//1024))
    return mask.astype(np.float64)

def rss_mib():
    return int(next(l.split()[1] for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('VmRSS:')))/1024

def benchmark_worker(algorithm,name,N):
    mask=scaled_mask(name,N);fn=exact_moments if algorithm=='tree' else exact_moments_direct
    baseline=rss_mib();start=time.perf_counter();mu,cov=fn(mask,True);first=time.perf_counter()-start
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    # Timings are medians of 5 single calls after one warm-up. Large direct
    # allocations stay isolated from the other implementation.
    times=[]
    for _ in range(5):
        start=time.perf_counter();fn(mask,True);times.append(time.perf_counter()-start)
    peak=max(peak,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    out=dict(algorithm=algorithm,mask=name,N=N,pixels=N*N,L=int(mask.sum()),
             median_ms=1000*float(np.median(times)),min_ms=1000*min(times),max_ms=1000*max(times),
             peak_RSS_MiB=peak,baseline_RSS_MiB=baseline,peak_minus_baseline_MiB=max(0.,peak-baseline),
             timing_calls=5,first_call_ms=first*1000,
             regime='native-derived decimation/padding' if N<=1024 else 'synthetic tiling of native-derived mask')
    print(json.dumps(out))

def benchmark():
    rows=[]
    for N in [128,256,512,1024,2048]:
        for name in ['footprint','stripes']:
            # Alternate order to avoid a systematic first-algorithm advantage.
            order=['direct','tree'] if (N.bit_length()+NAMES.index(name))%2 else ['tree','direct']
            for alg in order:
                p=subprocess.run([sys.executable,__file__,'--worker',alg,name,str(N)],capture_output=True,text=True,check=True)
                r=json.loads(p.stdout);rows.append(r);print('benchmark',N,name,alg,round(r['median_ms'],3),'ms',round(r['peak_RSS_MiB'],1),'MiB',flush=True)
    savecsv(RES/'benchmark.csv',rows)
    print('benchmark complete',flush=True)

def check_moments():
    rows=[]
    for N in [64,128,256,512,1024]:
        for name in NAMES:
            m=scaled_mask(name,N);mt,ct=exact_moments(m,True);md,cd=exact_moments_direct(m,True)
            rows.append(dict(N=N,mask=name,L=int(m.sum()),mean_max_abs=float(np.max(abs(mt-md))),
                             covariance_max_abs=float(np.max(abs(ct-cd))),
                             covariance_relative_Frobenius=float(np.linalg.norm(ct-cd)/np.linalg.norm(cd)),
                             sum_mean_error=abs(float(mt.sum())-1),covariance_sum_error=float(np.max(abs(ct.sum(axis=0)))),
                             min_cov_eigenvalue=float(np.linalg.eigvalsh(ct)[0])))
            assert np.max(abs(ct-cd))<1e-12
    savecsv(RES/'numerical_agreement.csv',rows)
    # A second arithmetic precision, with independent full-resolution details.
    extended=[]
    for N in [64,128,256]:
        for name in NAMES:
            mask=scaled_mask(name,N);m=mask.astype(np.longdouble);n=N*N;mm=N.bit_length()-1
            proj=[]
            for k in range(mm+1):
                side=N//2**k;g=2**k
                a=m.reshape(g,side,g,side).mean(axis=(1,3),dtype=np.longdouble)
                proj.append(np.repeat(np.repeat(a,side,0),side,1))
            dm=np.stack([proj[0]]+[proj[k]-proj[k-1] for k in range(1,len(proj))]).reshape(mm+1,-1)
            gg=(dm*m.ravel())@dm.T
            gt=mask_detail_gram_tree(mask);gd=(details(mask).reshape(mm+1,-1)*mask.ravel())@details(mask).reshape(mm+1,-1).T
            extended.append(dict(N=N,mask=name,tree_g_abs=float(np.max(abs(gt.astype(np.longdouble)-gg))),direct_g_abs=float(np.max(abs(gd.astype(np.longdouble)-gg)))))
    savecsv(RES/'extended_precision_agreement.csv',extended)
    # Profile moments on the actual 64x64 scene masks, with standard-error scales.
    rows=[];N=64;K=20000;rng=np.random.default_rng(20261034)
    for name in NAMES:
        mask=scaled_mask(name,N);mu,cov=exact_moments(mask,True);s=np.zeros(len(mu));ss=np.zeros_like(cov)
        for begin in range(0,K,128):
            z=rng.normal(size=(min(128,K-begin),N,N));p=profiles(preprocess(z,mask,True));s+=p.sum(0);ss+=p.T@p
        emp=s/K;ec=ss/K-np.outer(emp,emp);se=np.sqrt(np.diag(cov)/K);active=se>0
        rows.append(dict(mask=name,N=N,draws=K,max_mean_abs=float(np.max(abs(emp-mu))),
                         max_mean_SE=float(np.max(abs(emp[active]-mu[active])/se[active])),
                         covariance_relative_Frobenius=float(np.linalg.norm(ec-cov)/np.linalg.norm(cov))))
    savecsv(RES/'monte_carlo_moments.csv',rows)
    print('moment checks complete',flush=True)

def compute_stats(y,mask,mu,cov):
    p=profiles(y);h=.5*np.sum((np.sqrt(p)-np.sqrt(mu))**2,axis=-1)
    q=np.einsum('...i,ij,...j->...',p-mu,np.linalg.pinv(cov,rcond=1e-11),p-mu)
    peak=mask.sum()*np.max(y*y,axis=(-2,-1))/np.sum(y*y,axis=(-2,-1))
    return np.stack([h,q,peak],axis=-1)

def audit():
    N=64;rows=[]
    for idx,name in enumerate(NAMES):
        mask=scaled_mask(name,N);full=np.ones_like(mask);mu,cov=exact_moments(mask,True)
        for seed in SEEDS:
            rng=np.random.default_rng(np.random.SeedSequence([seed,64,idx,191]))
            matched=[];unmatched=[]
            for rep in range(R_AUDIT):
                z=rng.normal(size=(B+1,N,N));obs=compute_stats(preprocess(z[:1],mask,True),mask,mu,cov)
                ref=compute_stats(preprocess(z[1:],mask,True),mask,mu,cov)
                # Same statistic (including fixed mu,Gamma,L), deliberately wrong B.
                ref_bad=compute_stats(preprocess(z[1:],full,True),mask,mu,cov)
                matched.append(pvalues(obs,ref)[0]);unmatched.append(pvalues(obs,ref_bad)[0])
                if (rep+1)%50==0:print('audit',name,seed,rep+1,flush=True)
            matched=np.array(matched);unmatched=np.array(unmatched)
            np.savez_compressed(RES/f'audit_raw_{name}_{seed}.npz',matched=matched,unmatched=unmatched)
            for label,pv in [('matched',matched),('unmatched',unmatched)]:
                for k,meth in enumerate(METHODS):rows.append(row(pv[:,k],mask=name,seed=seed,reference=label,method=meth))
    savecsv(RES/'fresh_audit.csv',rows)
    print('audit complete',flush=True)

def power():
    N=64;rows=[]
    with np.load(DATA/'fixed_scene_N64.npz') as data:image=data['gray']
    for idx,name in enumerate(NAMES):
        mask=scaled_mask(name,N);mu,cov=exact_moments(mask,True);ell=mask.sum()-1
        clean=preprocess(image,mask,True);norm=np.linalg.norm(clean);assert norm>0;clean/=norm
        for seed in SEEDS:
            rng=np.random.default_rng(np.random.SeedSequence([seed,64,idx,192]))
            refs=compute_stats(preprocess(rng.normal(size=(B,N,N)),mask,True),mask,mu,cov)
            base_noise=preprocess(rng.normal(size=(R_POWER,N,N)),mask,True);raw={}
            for rho in RHOS:
                obs=base_noise+np.sqrt(rho*ell)*clean;st=compute_stats(obs,mask,mu,cov);pv=pvalues(st,refs)
                raw[f'pvalues_{rho:g}']=pv;raw[f'statistics_{rho:g}']=st
                for j,meth in enumerate(METHODS):rows.append(row(pv[:,j],mask=name,seed=seed,rho=rho,method=meth))
            raw['references']=refs;raw['clean_unit']=clean;raw['null_mean']=mu
            np.savez_compressed(RES/f'power_raw_{name}_{seed}.npz',**raw)
            print('power',name,seed,'complete',flush=True)
    savecsv(RES/'power.csv',rows)

def signal_moments():
    """Known signal-plus-noise raw energy moments on the real spatial template."""
    N=64;K=20000;rho=.1;rng=np.random.default_rng(20261035);rows=[]
    with np.load(DATA/'fixed_scene_N64.npz') as data:image=data['gray']
    for name in NAMES:
        mask=scaled_mask(name,N);mu,cov=exact_moments(mask,True);ell=mask.sum()-1
        clean=preprocess(image,mask,True);clean*=np.sqrt(rho*ell)/np.linalg.norm(clean)
        a=ell*mu
        # Recover Hc from the projected-sphere formula; the linear term is
        # formed independently in the spatial detail domain.
        hc=cov*(ell*(ell+2)/2)+np.outer(a,a)/ell
        dc=details(clean);energy=np.sum(dc*dc,axis=(1,2));v=preprocess(dc,mask,True).reshape(len(mu),-1)
        predicted_mean=energy+a;predicted_cov=2*hc+4*(v@v.T)
        ss=np.zeros_like(cov);sums=np.zeros_like(mu)
        from mask_profiles import energies
        for start in range(0,K,128):
            z=rng.normal(size=(min(128,K-start),N,N));e=energies(clean+preprocess(z,mask,True));sums+=e.sum(0);ss+=e.T@e
        empirical_mean=sums/K;empirical_cov=ss/K-np.outer(empirical_mean,empirical_mean)
        se=np.sqrt(np.maximum(0,np.diag(predicted_cov))/K);active=se>1e-12
        np.savez_compressed(RES/f'signal_moments_{name}.npz',predicted_mean=predicted_mean,predicted_cov=predicted_cov,empirical_mean=empirical_mean,empirical_cov=empirical_cov,rho=rho,draws=K)
        rows.append(dict(mask=name,N=N,rho=rho,draws=K,max_mean_SE=float(np.max(abs(empirical_mean[active]-predicted_mean[active])/se[active])),covariance_relative_Frobenius=float(np.linalg.norm(empirical_cov-predicted_cov)/np.linalg.norm(predicted_cov))))
    savecsv(RES/'signal_moments.csv',rows)
    import scipy,matplotlib,PIL
    meta=json.loads((RES/'protocol_and_provenance.json').read_text())
    meta.update(signal_moment_check=dict(seed=20261035,rho=.1,draws_per_mask=K),
                noise_moment_check=dict(seed=20261034,draws_per_mask=20000),
                float64_significand_bits=np.finfo(np.float64).nmant+1,
                longdouble_significand_bits=np.finfo(np.longdouble).nmant+1,
                scipy=scipy.__version__,matplotlib=matplotlib.__version__,pillow=PIL.__version__,
                nodata_caveat='file-defined band-mask intersection; interior zeros can include 8-bit quantization, not a certified physical missingness map')
    (RES/'protocol_and_provenance.json').write_text(json.dumps(meta,indent=2))
    print('signal-plus-noise moments complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',nargs=3);p.add_argument('--stage',choices=['prepare','benchmark','moments','audit','power','signal_moments','all'],default='all');a=p.parse_args()
    if a.worker:benchmark_worker(a.worker[0],a.worker[1],int(a.worker[2]));raise SystemExit
    funcs=dict(prepare=prepare,benchmark=benchmark,moments=check_moments,audit=audit,power=power,signal_moments=signal_moments)
    for key,fn in funcs.items():
        if a.stage in ['all',key]:fn()
