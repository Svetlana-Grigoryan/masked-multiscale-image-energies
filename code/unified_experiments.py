"""Prospectively fixed additional experiments for the unified manuscript.

Historical CSV/NPZ records are never overwritten. Defaults: b=999, R=500,
10 independent banks per fixed template/mask and rho in {0,.1}. New Q+V
combination was suggested after the earlier results: these fresh streams are
validation of that new fixed rule, not retroactive preregistration.
"""
from pathlib import Path
import os,sys,json,csv,hashlib,time,subprocess,platform,argparse
import numpy as np
from scipy.stats import t
from scipy.linalg.blas import dgemm
from PIL import Image
R=Path(__file__).resolve().parents[1];O=R/'results/unified';O.mkdir(exist_ok=True)
sys.path.append(str(R/'landsat_pilot/code'))
from mask_profiles import exact_moments,exact_moments_direct,prepare_q_metric,profiles,preprocess,dimensions,mask_gram,mask_detail_gram_tree,energies
from run_pilot import scaled_mask

def save(name,rows):
 with (O/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def summarize(vals):
 a=np.asarray(vals,float);h=t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a));return dict(mean=float(a.mean()),sd=float(a.std(ddof=1)),low=float(a.mean()-h),high=float(a.mean()+h),banks=len(a))
def stats(y,mask,mu,qfun):
 p=profiles(y);h=.5*np.sum((np.sqrt(p)-np.sqrt(mu))**2,axis=-1);v=mask.sum()*np.max(y*y,axis=(-2,-1))/np.sum(y*y,axis=(-2,-1))
 return np.stack([h,qfun(p),v],axis=-1)
def sorted_refs(ref):return np.sort(ref,axis=0)
def ranks(obs,sref):
 b=len(sref);qs=[]
 for j in range(3):
  eps=1e-12*np.maximum(1.,abs(obs[:,j]));qs.append((1+b-np.searchsorted(sref[:,j],obs[:,j]-eps,side='left'))/(b+1))
 lo=(1+np.searchsorted(sref[:,0],obs[:,0]+1e-12*np.maximum(1.,abs(obs[:,0])),side='right'))/(b+1)
 return np.stack([qs[0],qs[1],qs[2],np.minimum(1,3*np.minimum(np.minimum(qs[0],lo),qs[2])),np.minimum(1,2*np.minimum(qs[1],qs[2]))],axis=-1)
METHODS=['H_upper','Q_upper','V_upper','H_tails_V','Q_V']
def template_cases():
 with np.load(R/'landsat_pilot/data/fixed_scene_N64.npz') as a:x=a['gray']
 cases=[(n,x,scaled_mask(n,64)) for n in ['footprint','stripes','elliptical_gaps']]
 rgb=np.asarray(Image.open(R/'data/new_templates/astronaut.png'),dtype=float);gray=np.einsum('ijk,k->ij',rgb,[.2126,.7152,.0722])/255.;ix=4+8*np.arange(64);gray=gray[np.ix_(ix,ix)]
 yy,xx=np.indices((64,64));circle=((xx+.5-32)/32)**2+((yy+.5-32)/32)**2<.65**2
 cases += [('NASA_full',gray,np.ones((64,64))),('NASA_circle',gray,circle.astype(float))]
 return cases

def validation():
 rows=[];raw={};cases=template_cases()
 for ci,(name,x,mask) in enumerate(cases):
  mu,cov=exact_moments(mask,True);qfun,diag=prepare_q_metric(mu,cov);clean=preprocess(x,mask,True);clean/=np.linalg.norm(clean)
  raw[name+'_mask']=mask;raw[name+'_template']=x
  for bank in range(10):
   rng=np.random.default_rng(np.random.SeedSequence([20261111,ci,bank]));refs=stats(preprocess(rng.normal(size=(999,64,64)),mask,True),mask,mu,qfun);noise=preprocess(rng.normal(size=(500,64,64)),mask,True);sref=sorted_refs(refs);raw[f'{name}_{bank}_ref']=refs
   for rho in [0.,.1]:
    obs=stats(noise+np.sqrt(rho*(mask.sum()-1))*clean,mask,mu,qfun);pv=ranks(obs,sref);raw[f'{name}_{bank}_{rho:g}_obs']=obs;raw[f'{name}_{bank}_{rho:g}_p']=pv
    for k,meth in enumerate(METHODS):rows.append(dict(case=name,bank=bank,rho=rho,method=meth,R=500,rejected=int(np.sum(pv[:,k]<=.05)),rate=float(np.mean(pv[:,k]<=.05))))
  print('new validation',name,flush=True)
 save('validation.csv',rows);np.savez_compressed(O/'validation_raw.npz',**raw)
 summ=[];diff=[]
 for name,_,_ in cases:
  for rho in [0.,.1]:
   data={m:np.array([r['rate'] for r in rows if r['case']==name and r['rho']==rho and r['method']==m]) for m in METHODS}
   for m,a in data.items():summ.append(dict(case=name,rho=rho,method=m,**summarize(a)))
   for a,b in [('Q_upper','H_upper'),('Q_V','H_tails_V'),('Q_V','Q_upper')]:diff.append(dict(case=name,rho=rho,contrast=a+' minus '+b,**summarize(data[a]-data[b])))
 save('validation_summary.csv',summ);save('paired_power_differences.csv',diff)
 # Requested paired intervals for the earlier 20-bank table (same data paired).
 old=list(csv.DictReader((R/'results/jmiv/independent_banks.csv').open()));p=[]
 for name in ['footprint','stripes','elliptical_gaps']:
  v={m:np.array([float(r['rate']) for r in old if r['mask']==name and r['method']==m]) for m in ['Q_upper','H_upper']}
  p.append(dict(mask=name,contrast='Q_upper minus H_upper',**summarize(v['Q_upper']-v['H_upper'])))
 save('historical_paired_differences.csv',p)
 # Second source provenance, exact frozen templates, no scikit-image dependency.
 pth=R/'data/new_templates/astronaut.png';meta=dict(source_url='https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/astronaut.png',documentation='https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.astronaut',credit='NASA, photograph of Eileen Collins; distributed in scikit-image 0.25.2',rights='Public domain according to scikit-image documentation; NASA source credit retained',sha256=hashlib.sha256(pth.read_bytes()).hexdigest(),stored_shape=[512,512,3],sampling='pixel centers 4+8k, k=0..63; no interpolation',grayscale_weights=[.2126,.7152,.0722],null='controlled added Gaussian noise after fixed mask centering',limitations='A second public photographic template, not a second satellite scene or a sensor-noise validation',seed_stream=[20261111],banks=10,b=999,R=500,rhos=[0.,.1])
 (R/'data/new_templates/provenance.json').write_text(json.dumps(meta,indent=2))

def high_precision():
 import mpmath as mp
 from fractions import Fraction as F
 mp.mp.dps=70;rows=[]
 with np.load(R/'results/jmiv/sensitivity_raw.npz') as data:
  for name in ['footprint','stripes','elliptical_gaps','L2','L3','L5','L17']:
   mu=data[name+'_mu'];cov=data[name+'_cov'];p=data[name+'_p'][:1099];ev,diag=prepare_q_metric(mu,cov);base=ev(p);scale=diag['scale'];dp=p-mu
   if scale:
    w,u=np.linalg.eigh((cov+cov.T)/(2*scale));ix=w>1e-11*w[-1];W=u[:,ix]/np.sqrt(scale*w[ix])[None,:]
   else:W=np.empty((len(mu),0))
   alt=np.sum(dgemm(1.,dp,W)**2,axis=-1) if W.shape[1] else np.zeros(len(p))
   long=np.sum((dp.astype(np.longdouble)@W.astype(np.longdouble))**2,axis=-1).astype(float)
   # Construct exact rational mask moments from exact dyadic tree quantities.
   mask=data[name+'_mask'];L=int(mask.sum());ell=L-1;N=len(mask);d=dimensions(N);e=energies(mask);g=mask_detail_gram_tree(mask);h=mask_gram(mask);a=[F(int(L*di),N*N)-F(float(ei))/L for di,ei in zip(d,e)]
   c=[[2*(F(float(h[i,j]))-2*F(float(g[i,j]))/L+F(float(e[i]))*F(float(e[j]))/(L*L)-a[i]*a[j]/ell)/(ell*(ell+2)) for j in range(len(d))] for i in range(len(d))]
   m=[ai/ell for ai in a];mpf=lambda x:mp.mpf(x.numerator)/x.denominator
   cm=mp.matrix([[mpf(v) for v in row] for row in c]);vals,vec=mp.eigsy(cm);mx=max(vals);active=[i for i in range(len(d)) if vals[i]>mp.mpf('1e-11')*mx] if mx>mp.mpf('1e-60') else []
   high=[]
   for pp in p:
    dd=[mp.mpf(float(v))-mpf(mi) for v,mi in zip(pp,m)];q=sum(sum(vec[k,i]*dd[k] for k in range(len(d)))**2/vals[i] for i in active);high.append(float(q))
   high=np.array(high);pv=lambda q:(1+np.sum(q[:999,None]>=q[None,999:]-1e-12*np.maximum(1.,abs(q[None,999:])),axis=0))/1000
   b=pv(base);variants=[alt,long,high];maxdr=max(float(np.max(abs(pv(q)-b))) for q in variants);changed=max(int(np.sum((pv(q)<=.05)!=(b<=.05))) for q in variants)
   batch=np.concatenate([ev(p[i:i+17]) for i in range(0,len(p),17)])
   assert np.allclose(base,batch,rtol=1e-13,atol=1e-13);assert changed==0
   rows.append(dict(mask=name,rank_float=W.shape[1],rank_high=len(active),mp_decimal_digits=70,profiles=len(p),high_relative_max=float(np.max(abs(high-base)/np.maximum(1.,abs(high)))),blas_relative_max=float(np.max(abs(alt-base)/np.maximum(1.,abs(base)))),longdouble_relative_max=float(np.max(abs(long-base)/np.maximum(1.,abs(base)))),maximum_rank_change=maxdr,changed_decisions=changed))
 save('high_precision.csv',rows)

def worker(alg,N,mode):
 # Timing begins after input arrays exist; imports and I/O are excluded.
 mask=scaled_mask('footprint',N);rng=np.random.default_rng(np.random.SeedSequence([20261112,N,0 if mode=='fresh' else 1]));obsn=1 if mode=='fresh' else 20;batch=32
 def resident(key):return int(next(l.split()[1] for l in Path('/proc/self/status').read_text().splitlines() if l.startswith(key+':')))/1024
 base=resident('VmRSS');start=time.perf_counter();t0=start;mu,cov=(exact_moments if alg=='tree' else exact_moments_direct)(mask,True);mom=time.perf_counter()-t0;t0=time.perf_counter();qfun,_=prepare_q_metric(mu,cov);factor=time.perf_counter()-t0
 generate=process=0.;chunks=[]
 for i in range(0,999,batch):
  t0=time.perf_counter();z=rng.normal(size=(min(batch,999-i),N,N));generate+=time.perf_counter()-t0;t0=time.perf_counter();chunks.append(stats(preprocess(z,mask,True),mask,mu,qfun));process+=time.perf_counter()-t0
 t0=time.perf_counter();refs=sorted_refs(np.concatenate(chunks));sort=time.perf_counter()-t0;t0=time.perf_counter();z=rng.normal(size=(obsn,N,N));generate+=time.perf_counter()-t0;t0=time.perf_counter();obs=stats(preprocess(z,mask,True),mask,mu,qfun);process+=time.perf_counter()-t0;t0=time.perf_counter();pv=ranks(obs,refs);rank=time.perf_counter()-t0;total=time.perf_counter()-start
 print(json.dumps(dict(N=N,mode=mode,algorithm=alg,b=999,observations=obsn,batch_size=batch,moments_ms=1000*mom,factor_ms=1000*factor,rng_ms=1000*generate,preprocess_statistics_ms=1000*process,sort_ms=1000*sort,ranks_ms=1000*rank,total_ms=1000*total,amortized_ms=1000*total/obsn,peak_MiB=resident('VmHWM'),baseline_MiB=base,ranks_sha256=hashlib.sha256(pv.tobytes()).hexdigest())))
def benchmark():
 rows=[]
 for N in [64,128,256]:
  for mode in ['fresh','shared']:
   for rep in range(3):
    for alg in ['tree','direct']:
     p=subprocess.run([sys.executable,str(Path(__file__).absolute()),'--worker',alg,str(N),mode],env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),capture_output=True,text=True,check=True);row=json.loads(p.stdout);row['process_replicate']=rep;rows.append(row)
   subset=[r for r in rows if r['N']==N and r['mode']==mode];assert len(set(r['ranks_sha256'] for r in subset))==1;print('workflow',N,mode,flush=True)
 save('workflow.csv',rows)
 meta=dict(python=platform.python_version(),numpy=np.__version__,os=platform.platform(),cpu=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),''),thread_policy='one BLAS thread; fresh separate worker for each timing',exclusions='imports, file I/O, input-mask construction excluded; moments, factorization, RNG, preprocessing, statistics, bank sorting and ranks included',scopes='full new test with b=999; shared bank amortized over exactly 20 observations; no whole-test O(n) speedup claim')
 (O/'environment.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(sys.argv[2],int(sys.argv[3]),sys.argv[4]);raise SystemExit
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['validation','precision','benchmark','all'],default='all');args=parser.parse_args()
 design=dict(B=999,banks=10,observations_per_bank=500,rhos=[0.,.1],methods=METHODS,seed_streams=[20261111,20261112],status='Fixed at implementation before executing the new experiments; Q+V was suggested after historical studies, not retrospectively preregistered')
 (O/'design.json').write_text(json.dumps(design,indent=2))
 for name,fn in [('validation',validation),('precision',high_precision),('benchmark',benchmark)]:
  if args.stage in [name,'all']:fn()
