#!/usr/bin/env python3
"""Read-only historical exposure audit; writes only isolated repair artifacts.
Reconstructs exact frozen binding graph without importing mutation-bearing scripts.
Exposure labels describe historical use; every P5 score cohort is already evaluated.
"""
import csv, gzip, hashlib, json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
B=Path('/Users/yuan/Documents/Codex/2026-09-08/jie/benchmark_step1')
OUT=ROOT/'results/repairs/20261001/exposure'


def rows(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',newline='') as f:
        return list(csv.DictReader(f,delimiter='\t'))


def write(path, data, cols):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,delimiter='\t',lineterminator='\n')
        w.writeheader(); w.writerows(data)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()


class UF:
    def __init__(self):self.p={}
    def find(self,x):
        self.p.setdefault(x,x)
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]];x=self.p[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.p[max(a,b)]=min(a,b)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    gm=rows(ROOT/'data/splits/group_map.tsv');man=rows(ROOT/'data/splits/split_manifest.tsv')
    old=rows(ROOT/'reports/knot_resplit/old_version/split_manifest.tsv')
    gm_by={r['sample_id']:r for r in gm}; cur={r['sample_id']:r for r in man};before={r['sample_id']:r for r in old}
    assert len(gm_by)==len(gm)==len(cur)==len(before)==4858
    assert set(gm_by)==set(cur)==set(before)
    fs=rows(ROOT/'data/curated/fold_switch_global.tsv')
    cluster={}
    with open(ROOT/'data/interim/p202/l2_cluster.tsv') as f:
        for l in f:
            rep,m=l.rstrip('\n').split('\t');assert m not in cluster;cluster[m]=rep
    uf=UF();edges=[]
    for r in gm:uf.find('G::'+r['group_id'])
    for rep in set(cluster.values()):uf.find('C::'+rep)
    def bind(a,b,why):uf.union(a,b);edges.append((a,b,why))
    # group_map incorporates same-UniProt, FS pair, sequence-cluster and structural edges.
    for e in rows(ROOT/'data/splits/fs_group_edges.tsv'):
        a,b=e['sample_a'],e['sample_b']
        assert gm_by[a]['group_id']==gm_by[b]['group_id'],e
    for r in fs:
        for side in ['a','b']:
            ep=f"EP_{r['pair_id']}_{side.upper()}__{r['pdb_'+side].lower()}{r['chain_'+side].upper()}"
            bind('G::'+gm_by[r['pair_id']]['group_id'],'C::'+cluster[ep],'FS_endpoint_sequence_cluster')
    for r in gm:
        for acc in filter(None,r['uniprots'].split(';')):
            if acc in cluster:bind('G::'+r['group_id'],'C::'+cluster[acc],'same_uniprot_background')
    endpoint_source=B/'manifests/fold_pair_endpoint_sequence_structure_summary.tsv'
    epseq={(r['pdb_id'].lower(),r['requested_chain'].upper()):r['observed_sequence'] for r in rows(endpoint_source)}
    bgseq=rows(ROOT/'data/curated/fold_switch_unlabeled_sequence.tsv.gz')
    sha_acc=defaultdict(list)
    for r in bgseq:sha_acc[r['sequence_sha256']].append(r['uniprot_accession'])
    for r in fs:
        for side in ['a','b']:
            seq=epseq[(r['pdb_'+side].lower(),r['chain_'+side].upper())]
            for acc in sha_acc.get(hashlib.sha256(seq.encode()).hexdigest(),[]):
                if acc in cluster:bind('G::'+gm_by[r['pair_id']]['group_id'],'C::'+cluster[acc],'FS_endpoint_exact_sha_background')
    for r in rows(ROOT/'data/curated/fold_switch_matched_controls.tsv'):
        if r['ratio']=='1:0' or r['control_accession'] in ('','-','NONE_ELIGIBLE'):continue
        bind('G::'+gm_by[r['positive_pair']]['group_id'],'C::'+cluster[r['control_accession']],'matched_control')
    # Resolve case differences via chain_list record IDs, reject any collisions.
    exact_pc={};lower_sid={}
    for sid,r in cur.items():
        if r['task_area']=='knot':
            key=sid.split(':',1)[1].lower();assert key not in lower_sid;lower_sid[key]=sid
    for r in rows(ROOT/'data/interim/p303/knot_chain_list.tsv'):
        key=(r['pdb'].lower(),r['chain']);assert key not in exact_pc;exact_pc[key]=lower_sid[r['record_id'].lower()]
    knot_edges=rows(ROOT/'data/splits/knot_structural_edges.tsv')
    assert len(knot_edges)==2998
    for e in knot_edges:
        s=[]
        for col in ['chain_a','chain_b']:
            pdb,ch=e[col].rsplit('_',1);s.append(exact_pc[(pdb.lower(),ch)])
        assert float(e['tm_min'])>=0.6
        bind('G::'+gm_by[s[0]]['group_id'],'G::'+gm_by[s[1]]['group_id'],'knot_structural_min06')
    members=defaultdict(list)
    for node in uf.p:members[uf.find(node)].append(node)
    ids={root:'COMP_'+hashlib.sha256('\n'.join(sorted(nodes)).encode()).hexdigest()[:20] for root,nodes in members.items()}
    assert len(set(ids.values()))==len(ids),'component fingerprint collision'
    sid_comp={sid:ids[uf.find('G::'+r['group_id'])] for sid,r in gm_by.items()}
    # Verify exact equivalence to saved background group-node partition; do not regenerate splits.
    bgmanifest=rows(ROOT/'data/splits/fs_l2_background_manifest.tsv.gz')
    from_saved=defaultdict(set);to_saved=defaultdict(set);splitsets=defaultdict(set)
    for r in bgmanifest:
        component=ids[uf.find('C::'+r['cluster_rep'])]
        from_saved[r['group_node']].add(component);to_saved[component].add(r['group_node'])
        splitsets[component].add(r['split'])
    for sid,c in sid_comp.items():splitsets[c].add(cur[sid]['split'])
    assert all(len(v)==1 for v in from_saved.values()) and all(len(v)==1 for v in to_saved.values())
    assert all(len(v)==1 for v in splitsets.values()),'binding component crosses current splits'
    assert len(ids)==68875,(len(ids),68875)
    assert all(uf.find(a)==uf.find(b) for a,b,_ in edges)
    # Extraction is inference only; combined with documented CV runs establishes prior model-selection use.
    km=rows(ROOT/'data/interim/p303/emb/knots_dev/extract_manifest.tsv')
    knot_input={r['name'] for r in km}
    expected_k={sid for sid,r in before.items() if r['task_area']=='knot' and r['split']=='development' and sid.split(':',1)[1] in {q['record_id'] for q in rows(ROOT/'data/curated/knots_sequences.tsv')}}
    assert knot_input==expected_k and len(knot_input)==674
    dm=rows(ROOT/'data/interim/p303/emb/disorder_dev/extract_manifest.tsv')
    disorder_input={'disorder:'+r['name'] for r in dm}
    assert all(before[sid]['split']=='development' for sid in disorder_input)
    # Reconstruct which old disorder folds really trained/scored; no inference-only promotion.
    states=defaultdict(dict)
    for r in rows(ROOT/'data/curated/disorder_masks.tsv'):
        if r['mask']=='1' and r['state'] in ('0','1'):
            for pos in range(int(r['start']),min(int(r['end']),1022)+1):
                states[r['disprot_id']][pos]=int(r['state'])
    old_fold_classes=defaultdict(Counter)
    for r in dm:old_fold_classes[before['disorder:'+r['name']]['dev_fold']].update(states[r['name']].values())
    active_folds=[]
    for fold,counts in old_fold_classes.items():
        train=sum((v for k,v in old_fold_classes.items() if k!=fold),Counter())
        if counts[0] and counts[1] and train[0] and train[1]:active_folds.append(fold)
    assert len(active_folds)==5
    assert all(any(before[sid]['dev_fold']!=f for f in active_folds) for sid in disorder_input)
    assert sum(sum(v.values()) for v in old_fold_classes.values())==200364
    prior=set(knot_input)|disorder_input
    # FS strict explicitly used for L1/L2 and region model selection; all were development.
    strict={r['pair_id'] for r in fs if r['target']=='1' and r['valid_mask']=='1'}
    assert len(strict)==10
    prior.update(strict)
    # Historical legacy split members have independent logged prior use.
    legacy={r['pair_id'] for r in rows(ROOT/'reports/fs_three_layer/exposure_scope_audit.tsv') if r['classification'] in ('v5_split_member','earlier_split_only')}
    prior.update(legacy)
    # Old L2 stage-A operational PU training/selection: manifest retained locally; inference alone not exposure.
    l2old=rows(ROOT/'data/interim/p303/emb/l2_stage_a/extract_manifest.tsv')
    old_l2_comps=set()
    for r in l2old:
        name=r['name']
        if name in cluster:old_l2_comps.add(ids[uf.find('C::'+cluster[name])])
        elif name in cur:old_l2_comps.add(sid_comp[name])
    # Stage-B actually fitted all old development representatives with operational PU zeros.
    # This is cross-task fitting, distinct from experimentally observed negative labels.
    old_bg=rows(ROOT/'reports/knot_resplit/old_version/fs_l2_background_manifest.tsv.gz')
    old_l2_stageA_comps=set(old_l2_comps)
    old_l2_stageB_reps={r['cluster_rep'] for r in old_bg if r['split']=='development' and not r['uniprot_accession'].startswith('EP_')}
    assert len(old_l2_stageB_reps)==47604
    old_l2_comps.update(ids[uf.find('C::'+rep)] for rep in old_l2_stageB_reps)
    priorcomp={sid_comp[x] for x in prior}|old_l2_comps
    old_supervised_comp={sid_comp[x] for x in prior}
    # Post-resplit stage-B fitted the complete development representative frame,
    # not only background components linked to labeled samples. Lock all global dev.
    currentdev={c for c,sps in splitsets.items() if 'development' in sps}
    # Current supervised development/evaluation are separate from previous-development history.
    mapping=[]
    for sid in sorted(cur):
        r=cur[sid];c=sid_comp[sid]
        mapping.append(dict(sample_id=sid,component_id=c,task_area=r['task_area'],group_id=r['group_id'],current_split=r['split'],old_split=before[sid]['split'],prior_dev_direct=int(sid in prior),prior_knot_cv_direct=int(sid in knot_input),prior_disorder_cv_direct=int(sid in disorder_input),component_prior_dev_exposed=int(c in priorcomp),component_prior_supervised_exposed=int(c in old_supervised_comp),component_prior_l2_operational_exposed=int(c in old_l2_comps),current_dev_component=int(c in currentdev),evidence_status='documented_cv_plus_exact_inputs' if sid in knot_input|disorder_input else ('legacy_or_FS_record' if sid in prior else 'no_direct_prior_dev_evidence')))
    write(OUT/'component_map.tsv',mapping,list(mapping[0]))
    print('component_map.tsv ready',flush=True)
    summary={'graph_components_global':len(ids),'graph_components_labeled':len(set(sid_comp.values())),'graph_nodes':len(uf.p),'graph_edges_added_by_reason':dict(Counter(x[2] for x in edges)),'graph_zero_cross_component_edge':True,'graph_current_split_binding_violations':0,'background_partition_equivalence':True,'old_knot_inputs':len(knot_input),'old_disorder_inputs':len(disorder_input),'legacy_FS_direct':len(legacy),'prior_L2_operational_components':len(old_l2_comps),'prior_L2_stageA_components':len(old_l2_stageA_comps),'prior_L2_stageB_representatives':len(old_l2_stageB_reps),'old_disorder_fold_class_counts':{k:dict(v) for k,v in old_fold_classes.items()},'old_disorder_active_folds':sorted(active_folds),'prior_supervised_components':len(old_supervised_comp),'prior_all_components':len(priorcomp),'current_global_development_components':len(currentdev)}
    evaluated=[];cohorts=[]
    score_sources=[('confirmation_knot','results/confirmation/CONF-KNOT-PRESENCE-P502-v1/scores.tsv','sample_id'),('holdout_knot','results/holdout/HOLD-KNOT-PRESENCE-P505-v1/scores.tsv','sample_id'),('confirmation_disorder','results/confirmation/CONF-DISORDER-RES-P502-v1/residue_scores.tsv.gz','protein'),('holdout_disorder','results/holdout/HOLD-DISORDER-RES-P505-v1/residue_scores.tsv.gz','protein')]
    for tag,path,namecol in score_sources:
        sc=rows(ROOT/path);by=defaultdict(list)
        for r in sc:
            sid=r[namecol] if namecol=='sample_id' else 'disorder:'+r[namecol]
            assert sid in cur;by[sid].append(r)
        comp_counts=Counter(sid_comp[sid] for sid in by)
        positive_sids=set();negative_sids=set()
        for sid,rr in sorted(by.items()):
            c=sid_comp[sid];ys={int(r['label']) for r in rr};positive_sids.update([sid] if 1 in ys else []);negative_sids.update([sid] if 0 in ys else [])
            evaluated.append(dict(cohort=tag,sample_id=sid,component_id=c,current_split=cur[sid]['split'],old_split=before[sid]['split'],score_rows=len(rr),observed_labels=';'.join(map(str,sorted(ys))),prior_dev_direct=int(sid in prior),prior_knot_cv_direct=int(sid in knot_input),component_prior_dev_exposed=int(c in priorcomp),component_prior_supervised_exposed=int(c in old_supervised_comp),component_prior_l2_operational_exposed=int(c in old_l2_comps),current_dev_component=int(c in currentdev),P5_result_exposed=1,posthoc_sensitivity_eligible=int(c not in priorcomp and c not in currentdev),fresh_preregistered=0))
        clean=[sid for sid in by if sid_comp[sid] not in priorcomp and sid_comp[sid] not in currentdev]
        cohorts.append(dict(cohort=tag,n_samples=len(by),n_components=len(comp_counts),n_positive_proteins=len(positive_sids),n_negative_proteins=len(negative_sids),direct_prior_dev=sum(sid in prior for sid in by),direct_old_knot_dev=sum(sid in knot_input for sid in by),component_prior_exposed=sum(sid_comp[sid] in priorcomp for sid in by),posthoc_sensitive_n=len(clean),posthoc_sensitive_components=len({sid_comp[sid] for sid in clean}),posthoc_positive_proteins=sum(sid in positive_sids for sid in clean),posthoc_negative_proteins=sum(sid in negative_sids for sid in clean),fresh_preregistered_n=0,max_component_sample_count=max(comp_counts.values())))
    write(OUT/'evaluated_sample_exposure.tsv',evaluated,list(evaluated[0]))
    write(OUT/'cohort_summary.tsv',cohorts,list(cohorts[0]));summary['cohorts']=cohorts
    # Objective forward-only migration preflight: all historically/currently-used component closure -> dev,
    # all existing P5 scored components have already been read, so no fresh confirmation replacement.
    evaluatedcomps={r['component_id'] for r in evaluated}
    fs_descriptive=rows(ROOT/'results/confirmation/CONF-FSL2-RANK-P502-v1/descriptive_pairs.tsv')
    fs_descriptive_comps={sid_comp[r['pair_id']] for r in fs_descriptive}
    assert len(fs_descriptive)==6
    evaluatedcomps.update(fs_descriptive_comps)
    summary['P5_FS_descriptive_read_pairs']=len(fs_descriptive)
    summary['P5_FS_descriptive_read_components']=len(fs_descriptive_comps)
    force=priorcomp|currentdev|evaluatedcomps
    feasible=[]
    for sid,r in sorted(cur.items()):
        c=sid_comp[sid];oldsp=r['split'];newsp='development' if c in force else oldsp
        feasible.append(dict(sample_id=sid,component_id=c,current_split=oldsp,objective_future_split=newsp,migration_reason='historical_or_current_development' if c in priorcomp|currentdev else ('P5_evaluated_read' if c in evaluatedcomps else 'no_use_found'),task_area=r['task_area']))
    write(OUT/'future_migration_preflight.tsv',feasible,list(feasible[0]))
    eligible_knots={'knot:'+r['record_id'] for r in rows(ROOT/'data/curated/knots_sequences.tsv')}
    masks=rows(ROOT/'data/curated/disorder_masks.tsv')
    disorder_usable={'disorder:'+r['disprot_id'] for r in masks if r['mask']=='1' and r['state'] in ('0','1') and int(r['start'])<=1022}
    remaining=Counter()
    for r in feasible:
        if r['objective_future_split']=='development':continue
        area=r['task_area']
        if r['sample_id'] in eligible_knots or r['sample_id'] in disorder_usable:
            remaining[area+'|'+r['objective_future_split']]+=1
    summary['objective_migration_remaining_sequence_usable_knot_or_domain_usable_disorder']=dict(remaining)
    assert not remaining,'Unexpected unused native evaluation data requires separate audit'
    component_policy=[]
    for c in sorted(ids.values()):
        current_sp=next(iter(splitsets[c]))
        component_policy.append(dict(component_id=c,current_split=current_sp,force_future_development=int(c in force),historical_supervised=int(c in old_supervised_comp),historical_PU_operational=int(c in old_l2_comps),current_development=int(c in currentdev),P5_evaluated_read=int(c in evaluatedcomps-fs_descriptive_comps),P5_descriptive_read=int(c in fs_descriptive_comps),fresh_preregistered=0))
    assert all(r['force_future_development']==1 for r in component_policy if r['current_split']=='development')
    write(OUT/'component_future_policy.tsv',component_policy,list(component_policy[0]))
    # Background membership is needed for a global future split guard, including matched controls.
    import io
    with open(OUT/'background_component_map.tsv.gz','wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as z:
            with io.TextIOWrapper(z,encoding='utf-8',newline='') as f:
                w=csv.writer(f,delimiter='\t',lineterminator='\n')
                w.writerow(['uniprot_accession','cluster_rep','component_id','current_split','historical_PU_operational','force_future_development'])
                for r in sorted(bgmanifest,key=lambda r:r['uniprot_accession']):
                    c=ids[uf.find('C::'+r['cluster_rep'])]
                    w.writerow([r['uniprot_accession'],r['cluster_rep'],c,r['split'],int(c in old_l2_comps),int(c in force)])
    scope_inventory=[dict(cohort=tag,source=path,status='P5_result_already_read',native_component_mapped='yes',future_fresh_claim='no') for tag,path,_ in score_sources]
    scope_inventory += [dict(cohort='confirmation_FS_descriptive',source='results/confirmation/CONF-FSL2-RANK-P502-v1/descriptive_pairs.tsv',status='descriptive_scores_and_tier_already_read_no_strict_positive',native_component_mapped='yes',future_fresh_claim='no'),dict(cohort='holdout_FS',source='results/holdout/first_read_record.json',status='not_run_no_strict_positive',native_component_mapped='yes',future_fresh_claim='no'),dict(cohort='P507_type_LCO',source='results/p507_type_probe_v2/lco_predictions.tsv',status='posthoc_LCO_predictions_already_read',native_component_mapped='partial_new_sources_need_their_graph',future_fresh_claim='no'),dict(cohort='P508_external_disorder',source='results/p508_disorder_contrast/per_protein.tsv',status='external_results_already_read_label_scope_and_overlap_need_separate_audit',native_component_mapped='no_external_graph_not_fabricated',future_fresh_claim='no')]
    write(OUT/'P5_scope_inventory.tsv',scope_inventory,list(scope_inventory[0]))
    summary['objective_migration_counts']=dict(Counter(r['task_area']+'|'+r['objective_future_split'] for r in feasible))
    events=[
        dict(event='prior_KNOT_presence_and_type_CV',time_scope='2026-09-25 01:12 至 16:45 前；逐样本运行分钟未留档',n_samples=674,content='真实 presence/type 标签参与拟合和开发集交叉验证选层选C',evidence='data/interim/p303/emb/knots_dev/extract_manifest.tsv;reports/knot_binding_decision_point.md;reports/tasks/P3.03.md;scripts/run_probes_p303.py',certainty='high_cohort_reconstruction_individual_fit_timestamp_unavailable',influenced_method_selection='yes'),
        dict(event='prior_DISORDER_CV',time_scope='2026-09-25 06:27 左右日志留档；文件mtime只作辅助',n_samples=2278,content='有标签残基拟合及三层四C网格；重建五折均含双类且均有效',evidence='data/interim/p303/emb/disorder_dev/extract_manifest.tsv;results/probes_disorder_run.log;data/curated/disorder_masks.tsv;old split_manifest',certainty='high_log_plus_inputs_fold_reconstruction',influenced_method_selection='yes'),
        dict(event='prior_L2_stageA_PU_selection',time_scope='2026-09-25 重划分之前；精确逐样本分钟未留档',n_samples=10000,content='未标注背景临时编码为PU operational 0参与读取器拟合和选择，非可靠阴性',evidence='data/interim/p303/emb/l2_stage_a/extract_manifest.tsv;scripts/run_probes_p303.py;reports/tasks/P3.03.md',certainty='documented_operational_fit_distinct_from_observed_label',influenced_method_selection='yes_operational'),
        dict(event='prior_L2_stageB_PU_fit',time_scope='2026-09-25 16:45 决策点前已有全宇宙结果',n_samples=47604,content='旧development所有代表簇加strict病例直推拟合并传播，非仅表示前向',evidence='old_version/fs_l2_background_manifest.tsv.gz;reports/tasks/P3.03.md;scripts/run_l2_stage_b.py;用户返回stage-B日志',certainty='documented_operational_fit_scope_not_biological_negative',influenced_method_selection='fit_yes_cross_task_operational'),
        dict(event='curation_or_inference_only',time_scope='P1导入至结构补算',n_samples='not_added_as_supervised_fit',content='来源导入、序列前向、按结构停止、分配后标签构成计数；单靠此类读取不推断模型选择',evidence='reports/knot_resplit/step2_exposure_audit.md;logs/exposure_log.tsv',certainty='purpose_documented_selection_no_for_these_events',influenced_method_selection='no'),
        dict(event='P5_confirm_label_evaluation',time_scope='2026-09-27 14:03/14:04 Asia/Shanghai',n_samples='148 knot/418 disorder',content='冻结读取器指标计算；现已读结果；前置未见性历史审计失败',evidence='results/confirmation/first_read_record.json',certainty='exact_first_read_record',influenced_method_selection='not_proven_but_no_future_fresh_claim'),
        dict(event='P5_holdout_label_evaluation',time_scope='2026-09-30 19:50 Asia/Shanghai',n_samples='108 knot/494 disorder',content='最终保留指标计算；现已读结果',evidence='results/holdout/first_read_record.json',certainty='exact_first_read_record',influenced_method_selection='not_proven_but_no_future_fresh_claim')]
    write(OUT/'use_evidence.tsv',events,list(events[0]))
    # Source hashes permit audit rerun with exact immutable inputs.
    sources=['scripts/audit_p5_historical_exposure.py','data/splits/group_map.tsv','data/splits/fs_group_edges.tsv','data/splits/split_manifest.tsv','data/splits/knot_structural_edges.tsv','data/splits/fs_l2_background_manifest.tsv.gz','reports/knot_resplit/old_version/split_manifest.tsv','data/interim/p202/l2_cluster.tsv','data/interim/p303/knot_chain_list.tsv','data/curated/fold_switch_global.tsv','data/curated/fold_switch_unlabeled_sequence.tsv.gz','data/curated/fold_switch_matched_controls.tsv','data/curated/knots_sequences.tsv','results/probes_disorder_run.log','reports/knot_binding_decision_point.md','reports/tasks/P3.03.md','scripts/run_probes_p303.py','scripts/run_l2_stage_b.py','results/confirmation/first_read_record.json','results/holdout/first_read_record.json','results/confirmation/CONF-FSL2-RANK-P502-v1/descriptive_pairs.tsv','data/curated/disorder_masks.tsv','reports/knot_resplit/old_version/fs_l2_background_manifest.tsv.gz','reports/fs_three_layer/exposure_scope_audit.tsv','data/interim/p303/emb/knots_dev/extract_manifest.tsv','data/interim/p303/emb/disorder_dev/extract_manifest.tsv','data/interim/p303/emb/l2_stage_a/extract_manifest.tsv']+[p for _,p,_ in score_sources]
    write(OUT/'input_checksums.tsv',[{'path':p,'sha256':sha(ROOT/p)} for p in sources]+[{'path':str(endpoint_source),'sha256':sha(endpoint_source)}],['path','sha256'])
    with open(OUT/'qc.json','w') as f:json.dump(summary,f,indent=2,ensure_ascii=False,sort_keys=True);f.write('\n')
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
