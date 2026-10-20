#!/usr/bin/env python3
"""Produce manuscript tables and figures directly from the saved result files."""
from pathlib import Path
from collections import defaultdict
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta
from mask_profiles import *

ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results';FIG=ROOT/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42})
COLORS=['#215c94','#cb5b32','#6c5797','#27866c','#272d38']
LABELS=['H upper','H two-sided','Q upper','Spatial peak','Combined']


def load(name):
    with (RES/(name+'.csv')).open() as f:return list(csv.DictReader(f))


def select(rows,**kw):return [r for r in rows if all(str(r[k])==str(v) for k,v in kw.items())]


def interval(k,R):
    lo=0 if k==0 else beta.ppf(.025,k,R-k+1)
    hi=1 if k==R else beta.ppf(.975,k+1,R-k)
    return k/R,lo,hi


def cell(rows,ci=True):
    assert rows
    k=sum(int(r['rejected']) for r in rows);R=sum(int(r['R']) for r in rows)
    p,lo,hi=interval(k,R)
    return f'{p:.3f} [{lo:.3f},{hi:.3f}]' if ci else f'{p:.3f}'


def table(name,caption,label,header,rows,align):
    s='\\begin{table}[htbp]\\centering\\footnotesize\n'
    s+='\\setlength{\\tabcolsep}{3pt}\n'
    s+=f'\\caption{{{caption}}}\\label{{{label}}}\n'
    s+=f'\\begin{{tabular}}{{{align}}}\\toprule\n'
    s+=' & '.join(header)+r'\\\midrule'+'\n'
    s+='\n'.join(' & '.join(row)+r'\\' for row in rows)
    s+='\n\\bottomrule\\end{tabular}\n\\end{table}\n'
    (RES/(name+'.tex')).write_text(s)


def tables():
    a=load('fresh_audit');rows=[]
    cond=[(16,x,0) for x in ['full','circle','irregular','half','checker','random','one_pixel']]
    cond += [(N,'circle',1) for N in [16,32,64]]
    for N,mask,cen in cond:
        z=select(a,N=N,mask=mask,centered=cen)
        rows.append([str(N),mask.replace('_',' ')+(' + center' if cen else ''),
            cell(select(z,reference='unmatched',method='H_upper')),
            cell(select(z,reference='matched',method='H_upper')),
            cell(select(z,reference='matched',method='combined'),False)])
    table('audit_table',r'Fresh-reference null rejection at level $0.05$: $R=1200$ and $b=999$ per condition. Brackets are unconditional exact 95\% binomial intervals. Combined-test intervals and seed-specific results are supplied in Online Resource 1.',
          'tab:audit',['$N$','Mask / preprocessing','Unmatched $H_+$','Matched $H_+$','Combined'],rows,'rlllr')
    p=load('power');rows=[]
    for pattern in ['step','checker','bump','spike','phantom']:
        for rho in [.25,1.,4.]:
            z=select(p,N=32,mask='circle',seed=20261018,pattern=pattern,rho=rho)
            rows.append([pattern,f'{rho:g}']+[cell(select(z,method=m),False) for m in METHODS[:4]]+
                        [cell(select(z,method='combined'))])
    table('power_table',r'Power at $N=32$ with circular masked centering, first seed (20261018), $R=1000$, $b=999$. Here $\rho=\Lambda/(L-1)$. Brackets are exact 95\% intervals conditional on the reference bank. All methods\textquotesingle{} intervals and three-seed results are supplied.',
          'tab:power',['Signal',r'$\rho$',r'$H_+$',r'$H_2$',r'$Q_+$',r'$V_+$','Combined'],rows,'lrrrrrl')
    r=load('robustness');rows=[]
    for model,kind in [('gaussian','identity'),('spherical_t5','identity'),('iid_t5','identity'),
                       ('iid_laplace','identity'),('hetero','hetero'),('hetero','identity'),
                       ('smooth','smooth'),('smooth','identity')]:
        z=select(r,seed=20261018,model=model,reference_map=kind)
        modelname={'gaussian':'Gaussian','spherical_t5':r'Spherical $t_5$','iid_t5':r'iid $t_5$',
                   'iid_laplace':'iid Laplace','hetero':'Heteroscedastic','smooth':'Smoothed'}[model]
        rows.append([modelname,'included' if kind!='identity' else 'white']+
                    [cell(select(z,method=m)) for m in ['H_upper','H_two','combined']])
    table('robustness_table',r'Noise-model sensitivity at $N=32$ with centered circular masking. First seed, $R=3000$, $b=999$; intervals are conditional on each reference bank. Included means that the known extra map is applied to references as well as observations; white omits it.',
          'tab:robustness',['Noise','Reference',r'$H_+$',r'$H_2$','Combined'],rows,'lllll')
    r=load('permutation');rows=[]
    for model in ['gaussian','iid_t5','iid_laplace']:
        z=select(r,model=model)
        rows.append([model.replace('_',r'\_')]+[cell(select(z,reference=ref,method=m),ci=(m=='H_two'))
            for m in ['H_two','combined'] for ref in ['Gaussian_MC','retained_permutation']])
    table('permutation_table',r'Fresh-reference comparison at $N=16$, centered circular mask, $R=3000$ pooled independent trials per model, $b=999$. Brackets are unconditional exact 95\% binomial intervals. MC means Gaussian simulation; perm. means retained-pixel permutation.',
          'tab:permutation',['Noise',r'MC $H_2$',r'Perm. $H_2$','MC comb.','Perm. comb.'],rows,'lllll')
    t=load('timings');rows=[]
    for N in [16,32,64,128,256]:
        rows.append([str(N)]+[f"{float(select(t,N=N,operation=op)[0]['median_ms']):.3f}"
            for op in ['profile','uncentered_moments','centered_moments']])
    table('timing_table','Median milliseconds per call; implementation-specific measurements, not asymptotic speed claims.',
          'tab:timing',['$N$','Profile','Uncentered moments','Centered moments'],rows,'rrrr')
    c=json.loads((RES/'algebra_checks.json').read_text())
    def sci(v):
        a,b=f'{v:.2e}'.split('e')
        return a+r'\times10^{'+str(int(b))+'}'
    (RES/'algebra_summary.tex').write_text(
        'The maximum absolute errors were '+
        f"${sci(c['walsh_haar'])}$ for Walsh--Haar energies, ${sci(c['trace_gram'])}$ for the mask trace Gram matrix, "+
        f"${sci(c['centered_gram'])}$ for the centered normalized covariance, "+
        f"${sci(c['commutator'])}$ for the commutator norm, and ${sci(c['product'])}$ for the product identity. "+
        f"The largest simulated mean and covariance discrepancies were ${sci(c['mean_mc_error'])}$ and ${sci(c['covariance_mc_error'])}$, respectively.\n")


def figures():
    FIG.mkdir(exist_ok=True)
    N=32;masks=make_masks(N)
    fig=plt.figure(figsize=(7.1,4.5));gs=fig.add_gridspec(2,4,height_ratios=[1,1.1])
    for j,name in enumerate(['circle','half','checker','random']):
        ax=fig.add_subplot(gs[0,j]);ax.imshow(masks[name],cmap='Blues',vmin=0,vmax=1)
        ax.set_title(name.capitalize());ax.set_xticks([]);ax.set_yticks([])
    ax=fig.add_subplot(gs[1,:])
    for j,name in enumerate(['circle','half','checker','random']):
        mask=masks[name];vals=[]
        for r in range(6):
            g=2**r;s=N//g;theta=mask.reshape(g,s,g,s).mean(axis=(1,3))
            vals.append(np.max(np.sqrt(theta*(1-theta))))
        ax.plot(range(6),vals,'o-',label=name.capitalize(),color=COLORS[j])
    ax.set_xlabel('Partition level r');ax.set_ylabel('Exact transfer norm');ax.set_ylim(-.02,.54)
    ax.legend(ncol=4,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.17))
    fig.tight_layout();fig.savefig(FIG/'geometry.pdf');plt.close(fig)
    p=load('power')
    fig,axes=plt.subplots(2,3,figsize=(7.2,5.4),sharex=True,sharey=True)
    for ax,pat in zip(axes.ravel(),['step','checker','bump','spike','phantom']):
        z=select(p,N=32,mask='circle',pattern=pat)
        for j,method in enumerate(METHODS):
            zz=select(z,method=method);rho=sorted(set(float(r['rho']) for r in zz));vals=[]
            for x in rho:vals.append([float(r['rate']) for r in zz if float(r['rho'])==x])
            v=np.array(vals)
            ax.plot(rho,v.mean(1),label=LABELS[j],color=COLORS[j],lw=1.6)
            ax.fill_between(rho,v.min(1),v.max(1),color=COLORS[j],alpha=.12)
        ax.set_title(pat.capitalize());ax.set_xscale('symlog',linthresh=.05)
        ax.set_xticks([0,.1,1.,10.],labels=['0',r'$10^{-1}$','1','10']);ax.minorticks_off()
        ax.axhline(.05,color='gray',ls=':',lw=.7);ax.set_ylim(-.02,1.02)
        ax.set_xlabel('Energy ratio rho')
    for ax in axes[:,0]:ax.set_ylabel('Rejection fraction')
    handles,labels=axes[0,0].get_legend_handles_labels();axes[1,2].axis('off')
    axes[1,2].legend(handles,labels,frameon=False,loc='center')
    fig.tight_layout();fig.savefig(FIG/'power.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,3.4));z=select(p,N=16,mask='full',pattern='spike')
    for j,method in enumerate(METHODS):
        zz=select(z,method=method);rho=sorted(set(float(r['rho']) for r in zz))
        v=np.array([[float(r['rate']) for r in zz if float(r['rho'])==x] for x in rho])
        ax.plot(rho,v.mean(1),'o-',ms=3,label=LABELS[j],color=COLORS[j])
        ax.fill_between(rho,v.min(1),v.max(1),color=COLORS[j],alpha=.1)
    ax.set_xscale('symlog',linthresh=.05);ax.set_xticks([0,.1,1.,10.],labels=['0',r'$10^{-1}$','1','10']);ax.minorticks_off();ax.set_xlabel('Signal / expected noise energy, rho')
    ax.set_ylabel('Rejection fraction');ax.legend(frameon=False,ncol=2);ax.set_ylim(-.02,1.02)
    fig.tight_layout();fig.savefig(FIG/'impulse.pdf');plt.close(fig)
    N=64;mask=make_masks(N)['circle'];x=phantom(N);clean=preprocess(x,mask,True)
    rng=np.random.default_rng(420);noise=preprocess(rng.normal(size=(N,N)),mask,True)
    noisy=np.sqrt(.25*(mask.sum()-1))*clean/np.linalg.norm(clean)+noise
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.65))
    for ax,im,title in zip(axes,[x,clean,noisy],['Five-ellipse phantom','Masked and centered','Noisy, rho = 0.25']):
        ax.imshow(im,cmap='gray');ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
    fig.tight_layout();fig.savefig(FIG/'phantom.pdf');plt.close(fig)
    for N in [16,32,64]:
        np.savez_compressed(RES/f'fixed_arrays_N{N}.npz',**{f'mask_{k}':v for k,v in make_masks(N).items()},
                            **{f'signal_{k}':v for k,v in signals(N).items()})


if __name__=='__main__':tables();figures()
