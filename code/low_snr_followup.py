"""Exploratory low-SNR follow-up after saturation at rho=.1; separate streams."""
from unified_experiments import *
def run():
 rows=[];raw={}
 for ci,(name,x,mask) in enumerate(template_cases()[3:]):
  mu,cov=exact_moments(mask,True);qfun,_=prepare_q_metric(mu,cov);clean=preprocess(x,mask,True);clean/=np.linalg.norm(clean)
  for bank in range(10):
   rng=np.random.default_rng(np.random.SeedSequence([20261113,ci,bank]));ref=stats(preprocess(rng.normal(size=(999,64,64)),mask,True),mask,mu,qfun);noise=preprocess(rng.normal(size=(500,64,64)),mask,True);raw[f'{name}_{bank}_ref']=ref
   for rho in [.005,.01,.025]:
    obs=stats(noise+np.sqrt(rho*(mask.sum()-1))*clean,mask,mu,qfun);pv=ranks(obs,sorted_refs(ref));raw[f'{name}_{bank}_{rho:g}_obs']=obs;raw[f'{name}_{bank}_{rho:g}_p']=pv
    for i,meth in enumerate(METHODS):rows.append(dict(case=name,bank=bank,rho=rho,method=meth,R=500,rejected=int(np.sum(pv[:,i]<=.05)),rate=float(np.mean(pv[:,i]<=.05))))
  print('low-SNR follow-up',name,flush=True)
 save('nasa_low_snr.csv',rows);np.savez_compressed(O/'nasa_low_snr_raw.npz',**raw)
 summ=[]
 for name,_,_ in template_cases()[3:]:
  for rho in [.005,.01,.025]:
   for meth in METHODS:summ.append(dict(case=name,rho=rho,method=meth,**summarize([r['rate'] for r in rows if r['case']==name and r['rho']==rho and r['method']==meth])))
 save('nasa_low_snr_summary.csv',summ)
 (O/'low_snr_design.json').write_text(json.dumps(dict(reason='Separate follow-up added after rho=.1 saturation; all three new ratios retained, no best-ratio selection',stream=[20261113],rhos=[.005,.01,.025],banks=10,b=999,R=500),indent=2))
if __name__=='__main__':run()
