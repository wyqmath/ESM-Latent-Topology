import csv,gzip,json,hashlib,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.special import expit
ROOT=Path(sys.argv[1]); E=Path(sys.argv[2]); OUT=Path(sys.argv[3]); manifest=list(csv.DictReader(open(ROOT/'data/interim/p508_emb/extract_manifest.tsv'),delimiter='\t')); rows=list(csv.DictReader(gzip.open(E/'residue_scores.tsv.gz','rt'),delimiter='\t')); groups=defaultdict(list)
for r in rows:groups[r['name']].append(r)
coef=np.load(E/'frozen_reader_coefficients.npz',allow_pickle=False);prov=json.loads((E/'replay_provenance.json').read_text());errors=[];checked=0
for row in manifest:
 if row['name'] not in groups:continue
 cache=ROOT/'data/interim/p508_emb'/(row['key']+'.npz');data=np.load(cache,allow_pickle=False);meta=json.loads(data['meta'].item());assert meta['layer_set']=='mean_all34+resid_33|domidx'
 assert meta['model']=='facebook/esm2_t33_650M_UR50D' and meta['revision']=='08e4846e537177426273712802403f7ba8261b6c'
 assert hashlib.sha256(cache.read_bytes()).hexdigest()==prov['representation_sha256'][str(cache)]
 g=groups[row['name']];assert data['resid_indices'].tolist()==[int(r['sequence_index_0based'])+1 for r in g]
 probability=expit(data['resid_layers'][0].astype(np.float32)@coef['coef'][0]+coef['intercept'][0]);expected=np.array([float(r['score']) for r in g]);delta=float(np.max(abs(probability-expected)));assert delta<1e-13,(row['name'],delta)
 errors.append(delta);checked+=len(g)
assert checked==22646 and len(errors)==99
assert not OUT.exists();OUT.write_text(json.dumps({'status':'PASS','independent_coefficient_sigmoid_all99_caches':True,'n_residues':checked,'n_proteins':len(errors),'max_abs_score_error':max(errors),'all99_cache_SHA_match_executed_provenance':True,'no_fit_no_representation_extraction':True,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n');print(OUT.read_text())
