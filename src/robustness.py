"""How sensitive is the val ranking of methods to the definition of a 'watched' target?"""
import numpy as np, pandas as pd
from core import *
import methods as M, hybrid as H
g, s, users = load()
raw = pd.read_parquet(f'{D}/user_airing.parquet'); raw['wk'] = 11 + (raw.air_t0 - WEEK0) // WEEK
folds = {n: Fold(n, g, s, users) for n in ['fit', 'val']}
data = {n: H.build(f) for n, f in folds.items()}
clf, feats = H.fit([data['fit'][0]])
fv = folds['val']; fb = M.pop_order(fv)
recs = {'hybrid': H.predict(clf, feats, data['val'][0], fv, fb)}
for m in H.SCORERS:
    recs[m] = topk(data['val'][1][m], fv.items, fb, fv.cold)
tg = raw[(raw.wk >= 25) & (raw.wk <= 30)]
defs = {'main: >=min(300s,50%)': watched(tg), 'any overlap >=60s': tg.sec >= 60, '>=50% of airing': tg.frac >= 0.5,
        '>=15 min': tg.sec >= 900}
rows = []
for dn, mask in defs.items():
    fv.truth = tg[mask].groupby('user_id').tv_show_id.agg(set).to_dict()
    rows.append({'target': dn, **{m: round(mapk(fv, r), 4) for m, r in recs.items()}})
print(pd.DataFrame(rows).to_string(index=False))
