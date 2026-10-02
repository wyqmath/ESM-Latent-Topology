#!/usr/bin/env python3
"""R5.05: recompute paired component bootstrap from immutable P4 OOF predictions.
No representation extraction, fitting, hyperparameter/window selection or split changes.
The estimand remains pooled AUROC/AUPRC; full binding components are sampled uniformly.
"""
import argparse,csv,gzip,hashlib,json,os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
from multiprocessing import get_context
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
B=2000;BOOT_SEEDS=[13,42,2026];TRAIN_SEEDS=[13,42,2026]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rd(p):
    op=gzip.open if str(p).endswith('.gz') else open
    with op(p,'rt',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def write(p,rows,cols):
    with open(p,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)

def score(p):
    rows=rd(p);out={}
    for r in rows:
        if r['unit'] in out:raise ValueError(f'duplicate unit in {p}: {r["unit"]}')
        out[r['unit']]=(int(r['y']),float(r['score']))
    if not all(y in (0,1) and np.isfinite(s) for y,s in out.values()):raise ValueError('bad prediction')
    return out

class WeightedMetric:
    def __init__(self,y,s,kind):
        self.y=np.asarray(y,dtype=int);self.kind=kind
        self.order=np.argsort(-np.asarray(s),kind='stable')
        desc=np.asarray(s)[self.order]
        self.starts=np.r_[0,np.flatnonzero(desc[1:]!=desc[:-1])+1]
    def __call__(self,w):
        w=np.asarray(w,dtype=float)
        tp=np.add.reduceat((w*self.y)[self.order],self.starts)
        fp=np.add.reduceat((w*(1-self.y))[self.order],self.starts)
        p,n=tp.sum(),fp.sum()
        if not p or not n:return float('nan')
        ctp=np.cumsum(tp);cfp=np.cumsum(fp)
        if self.kind=='auprc':
            precision=np.divide(ctp,ctp+cfp,out=np.zeros_like(ctp),where=(ctp+cfp)>0)
            return float(np.dot(tp,precision)/p)
        # Descending-score positives rank above every lower-scoring negative;
        # equal-score positive-negative pairs each count one half.
        below=n-cfp
        return float(np.dot(tp,below+fp/2)/(p*n))

def metric_self_test():
    tests=0
    rng=np.random.RandomState(117)
    for kind,ref in [('auroc',roc_auc_score),('auprc',average_precision_score)]:
        for i in range(100):
            y=rng.randint(0,2,20);s=rng.choice([.1,.2,.7,.9],20);w=rng.randint(0,5,20)
            if not w[y==1].sum() or not w[y==0].sum():continue
            got=WeightedMetric(y,s,kind)(w)
            expected=ref(y,s,sample_weight=w)
            expanded=ref(np.repeat(y,w),np.repeat(s,w))
            assert abs(got-expected)<1e-14 and abs(got-expanded)<1e-14
            tests+=1
    return tests

_JOB_DATA={}
def run_job(spec):
    tag,refname,armname,kind,nfam,bootseed=spec
    ref=_JOB_DATA[refname];arm=_JOB_DATA[armname]
    ids=sorted(ref)
    if set(ids)!=set(arm):raise ValueError(f'{tag}: row-universe mismatch')
    y=np.array([ref[u][0] for u in ids]);s0=np.array([ref[u][1] for u in ids]);s1=np.array([arm[u][1] for u in ids])
    if any(arm[u][0]!=ref[u][0] for u in ids):raise ValueError('paired labels differ')
    sid=[]
    for u in ids:
        protein=u.split('#',1)[0]
        sid.append(protein if kind=='auroc' else 'disorder:'+protein)
    component_map=_JOB_DATA['components']
    if any(x not in component_map for x in sid):raise ValueError(f'{tag}: missing authoritative component')
    components=[component_map[x]['component_id'] for x in sid]
    if any(component_map[x]['current_split']!='development' for x in sid):raise ValueError('OOF row is not current development')
    units=sorted(set(components));unit_idx={x:i for i,x in enumerate(units)}
    mem=np.array([unit_idx[x] for x in components]);rng=np.random.RandomState(bootseed)
    m0=WeightedMetric(y,s0,kind);m1=WeightedMetric(y,s1,kind)
    point0=m0(np.ones(len(ids)));point1=m1(np.ones(len(ids)))
    refmetric=roc_auc_score if kind=='auroc' else average_precision_score
    assert abs(point0-refmetric(y,s0))<1e-13 and abs(point1-refmetric(y,s1))<1e-13
    out=[];deltas=[]
    for b in range(B):
        counts=np.bincount(rng.randint(0,len(units),len(units)),minlength=len(units));w=counts[mem]
        a,bv=m0(w),m1(w);delta=bv-a
        out.append({'draw':b,'ref_metric':a,'arm_metric':bv,'delta':delta})
        if np.isfinite(delta):deltas.append(delta)
    v=np.sort(deltas);cut=.005/nfam
    # Preserve the original registered order-statistic convention, rather than changing to interpolated quantiles.
    interval=[None,None] if len(v)==0 else [float(v[int(np.floor(cut*len(v)))]),float(v[int(np.ceil((1-cut)*len(v)))-1])]
    enough=len(v)>=1000
    record={'comparison':tag,'ref':refname,'arm':armname,'metric':kind,'family_n':nfam,'bootstrap_seed':bootseed,'B':B,'n_rows':len(ids),'n_proteins':len(set(sid)),'n_components':len(units),'point_ref':point0,'point_arm':point1,'delta_point':point1-point0,'ci99fam':interval,'boot_valid':len(v),'boot_degenerate':B-len(v),'no_conclusion':not enough,'sig99fam':bool(enough and (interval[0]>0 or interval[1]<0)),'cut_each_tail':cut,'estimand':'pooled metric difference, ref and arm weighted by same component multiplicity','selection_uncertainty':'conditions on archived selected readers/windows, not retraining or reselection'}
    return record,out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fixed-sources',type=Path,default=ROOT/'results/repairs/20261001/aggregation/source_scores/fixed')
    p.add_argument('--learned-sources',type=Path,default=ROOT/'results/aggregation/learned')
    p.add_argument('--component-map',type=Path,default=ROOT/'results/repairs/20261001/exposure/component_map.tsv')
    args=p.parse_args()
    if args.output.exists() and list(args.output.iterdir()):p.error('output nonempty; refusing overwrite')
    args.output.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),args.component_map,ROOT/'configs/aggregation.yaml',ROOT/'configs/evaluation_protocol.yaml',ROOT/'results/aggregation/fixed/aggregation_fixed_qc.json',ROOT/'results/aggregation/learned/aggregation_learned_qc.json']
    comp=rd(args.component_map);_JOB_DATA['components']={r['sample_id']:r for r in comp};assert len(_JOB_DATA['components'])==len(comp)
    missing=[]
    for tag in ['D-ID','D-MEAN','D-LAST','D-WIN4','D-WIN16','D-WIN64','D-ID_h23','D-MEAN_h23','K-POOLED','K-LAST','K-WIN4','K-WIN16','K-WIN64','K-MIL_mean','K-MIL_max','K-POOLED_h29','K-LAST_h29','K-MIL_mean_h29']:
        path=args.fixed_sources/(tag+'.scores.tsv.gz')
        if not path.exists():missing.append(tag);continue
        _JOB_DATA[tag]=score(path);files.append(path)
    for task,label in [('knots','L-ATT-KNOTS'),('disorder','L-ATT-DISORDER')]:
        for seed in TRAIN_SEEDS:
            path=args.learned_sources/f'{task}_latt_seed{seed}.tsv.gz'
            tag=f'{label}_trainseed{seed}'
            if not path.exists():missing.append(tag);continue
            _JOB_DATA[tag]=score(path);files.append(path)
        tags=[f'{label}_trainseed{x}' for x in TRAIN_SEEDS]
        if all(t in _JOB_DATA for t in tags):
            ids=set(_JOB_DATA[tags[0]])
            if any(set(_JOB_DATA[t])!=ids for t in tags):raise ValueError('seed-mean IDs differ')
            mean={}
            for u in sorted(ids):
                yy=[_JOB_DATA[t][u][0] for t in tags]
                if len(set(yy))!=1:raise ValueError('seed-mean labels differ')
                mean[u]=(yy[0],float(np.mean([_JOB_DATA[t][u][1] for t in tags])))
            _JOB_DATA[label+'_seedmean']=mean
    oldqc=json.loads((ROOT/'results/aggregation/fixed/aggregation_fixed_qc.json').read_text())
    winD=oldqc['selection']['disorder_win']['selected'];winK=oldqc['selection']['knots_win']['selected']
    plans=[]
    def plan(tag,ref,arm,kind,nfam):
        if ref not in _JOB_DATA or arm not in _JOB_DATA:missing.append(tag);return
        plans.extend((tag,ref,arm,kind,nfam,s) for s in BOOT_SEEDS)
    for task,ref,arms,kind in [('fixed_global','K-POOLED',['K-MIL_mean','K-LAST',f'K-WIN{winK}'],'auroc'),('fixed_residue','D-ID',['D-MEAN','D-LAST',f'D-WIN{winD}'],'auprc')]:
        for arm in arms:plan(f'{task}::{arm}_vs_{ref}',ref,arm,kind,3)
    for arm in ['K-WIN4','K-WIN16','K-WIN64','K-MIL_max','K-POOLED_h29','K-LAST_h29','K-MIL_mean_h29']:
        if arm==f'K-WIN{winK}':continue
        plan(f'sensitivity::{arm}_vs_K-POOLED','K-POOLED',arm,'auroc',3)
    for arm in ['D-WIN4','D-WIN16','D-WIN64']:
        if arm==f'D-WIN{winD}':continue
        plan(f'sensitivity::{arm}_vs_D-ID','D-ID',arm,'auprc',3)
    plan('sensitivity::D-ID_h23_vs_D-ID','D-ID','D-ID_h23','auprc',3)
    plan('sensitivity::D-MEAN_h23_vs_D-MEAN','D-MEAN','D-MEAN_h23','auprc',3)
    for label,ref,kind in [('L-ATT-KNOTS','K-POOLED','auroc'),('L-ATT-DISORDER','D-MEAN','auprc')]:
        for ts in TRAIN_SEEDS:plan(f'learned_seedwise::{label}_trainseed{ts}_vs_{ref}',ref,f'{label}_trainseed{ts}',kind,2)
        plan(f'learned_primary::{label}_seedmean_vs_{ref}',ref,label+'_seedmean',kind,2)
    tests=metric_self_test();records=[]
    # Hardware-available CPU workers, no requested compute quota. Fork shares immutable score arrays.
    with ProcessPoolExecutor(max_workers=os.cpu_count(),mp_context=get_context('fork')) as pool:
        for rec,draws in pool.map(run_job,plans):
            fn=rec['comparison'].replace('::','__')+'_bootseed'+str(rec['bootstrap_seed'])+'.tsv'
            write(args.output/fn,draws,['draw','ref_metric','arm_metric','delta'])
            records.append(rec);print(rec['comparison'],rec['bootstrap_seed'],rec['delta_point'],rec['ci99fam'],flush=True)
    compact=[{k:r[k] for k in ['comparison','bootstrap_seed','metric','family_n','n_rows','n_proteins','n_components','point_ref','point_arm','delta_point','boot_valid','boot_degenerate','sig99fam','no_conclusion']}|{'ci99fam_low':r['ci99fam'][0],'ci99fam_high':r['ci99fam'][1]} for r in records]
    if compact:write(args.output/'summary.tsv',compact,list(compact[0]))
    meta={'created_at_utc':datetime.now(timezone.utc).isoformat(),'kind':'component-bootstrap correction of archived development OOF scores','n_records':len(records),'bootstrap_parameters':{'B':B,'seeds':BOOT_SEEDS,'family_size_fixed':3,'family_size_learned':2,'each_tail':'.005/NF','CI_method':'original registered discrete order statistics'},'archived_window_selection':{'disorder':winD,'knots':winK},'weighted_metric_self_tests':tests,'missing_unrecomputable':missing,'records':records,'input_sha256':{str(x.relative_to(ROOT)):sha(x) for x in files},'limitations':['Reader/C/window selection fixed; bootstrap does not cover development selection or training uncertainty.','Shared OOF training introduces cross-fold dependencies not eliminated by component resampling; intervals conditional on archived predictions.','Learned seed-mean scores averaged by identical row IDs before calculating pooled metric.','Mean of seed metrics is not the registered primary estimand.','Scores archived with six significant digits; point estimates from these scores can differ slightly from original unrounded metrics.','Binding components do not imply biological families or max-TM independence.']}
    (args.output/'aggregation_statistics.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Complete records',len(records),'selftests',tests,flush=True)

if __name__=='__main__':main()
