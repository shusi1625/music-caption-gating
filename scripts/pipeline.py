"""CPU-only frozen development calibration and ranking primitives."""
from pathlib import Path
import hashlib, json
from fractions import Fraction
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()

def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    tmp.replace(path)

def canonical_hash(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def normalize(x):
    norms=np.linalg.norm(x,axis=-1,keepdims=True)
    if not np.isfinite(x).all() or np.any(norms<1e-12):raise ValueError('Invalid embeddings')
    return x/norms

def rank_metrics(scores,catalog_ids,source_ids):
    catalog_ids=list(map(str,catalog_ids)); source_ids=list(map(str,source_ids))
    scores=np.asarray(scores)
    if scores.shape!=(len(source_ids),len(catalog_ids)) or not np.isfinite(scores).all():
        raise ValueError('Score shape or values invalid')
    if len(set(catalog_ids))!=len(catalog_ids):raise ValueError('Duplicate catalog IDs')
    columns=sorted(range(len(catalog_ids)),key=lambda j:int(catalog_ids[j]))
    ordered_ids=np.array(catalog_ids)[columns]
    if not set(source_ids)<=set(catalog_ids):raise ValueError('Source outside catalog')
    ordering=np.argsort(-scores[:,columns],axis=1,kind='stable')
    ranks=np.argmax(ordered_ids[ordering]==np.array(source_ids)[:,None],axis=1)+1
    return {'MRR':float(np.mean(1/ranks)),'R1':float(np.mean(ranks<=1)),
            'R5':float(np.mean(ranks<=5)),'R10':float(np.mean(ranks<=10)), 'ranks':ranks.tolist()}

def calibrate(audio,index_text,query_text,catalog_ids,source_ids,lambda_grid):
    # All arrays must be development data. No evaluation metrics are calculated here.
    audio=np.asarray(audio);index_text=np.asarray(index_text);query_text=np.asarray(query_text)
    if audio.ndim!=3 or audio.shape[1]!=12:raise ValueError('Need twelve fixed windows')
    mean=normalize(normalize(audio).mean(axis=1))
    index_text=normalize(index_text);query_text=normalize(query_text)
    views={'first_window':normalize(audio[:,0]),'normalized_12_window_mean':mean}
    score={k:np.sum(v*index_text,axis=1) for k,v in views.items()}
    thresholds={k:float(np.quantile(v,.1,method='linear')) for k,v in score.items()}
    retain={k:float(np.mean(v>=thresholds[k])) for k,v in score.items()}
    A=query_text@mean.T;T=query_text@index_text.T
    grid=[];best=None;selected=None
    for weight in sorted(lambda_grid):
        m=rank_metrics((1-weight)*A+weight*T,catalog_ids,source_ids)
        exact=sum((Fraction(1,r) for r in m['ranks']),Fraction(0))/len(source_ids)
        grid.append({'lambda':weight,**m,'exact_mrr':str(exact)})
        if best is None or exact>best:best=exact;selected=weight
    return {'scope':'development_only','thresholds':thresholds,'retained_fraction':retain,
            'threshold_input_count':len(catalog_ids),'query_count':len(source_ids),
            'lambda_grid_results':grid,'selected_lambda':selected,
            'H2_useful_caption_premise': 'unsupported' if selected==0 else 'development_support_only',
            'development_scores':{k:v.tolist() for k,v in score.items()}}

def load_development(manifest,cache_dir):
    items=sorted([r for r in manifest if r['split']=='development'],key=lambda r:int(r['track_id']))
    audio=[];index=[];queries=[];catalog=[];sources=[]
    for item in items:
        tid=str(item['track_id'])
        with np.load(Path(cache_dir)/(tid+'.npz'),allow_pickle=False) as data:
            audio.append(data['audio']);index.append(data['text'][0]);catalog.append(tid)
            if len(item['captions'])>=2:
                queries.append(data['text'][1]);sources.append(tid)
    return dict(audio=np.stack(audio),index_text=np.stack(index),query_text=np.stack(queries),
                catalog_ids=catalog,source_ids=sources)

