"""Val MAP@5 by viewer-activity tercile, and familiarity/novelty of each method's Top-5 (val and test)."""
import pickle, numpy as np, pandas as pd
from core import *
import methods as M, hybrid as H
g, s, users = load()
ff, fv = Fold('fit', g, s, users), Fold('val', g, s, users)
Xf, _ = H.build(ff); Xv, scv = H.build(fv); fb = M.pop_order(fv)
recs = {m: topk(scv[m], fv.items, fb, fv.cold) for m in H.SCORERS}
clf, feats = H.fit([Xf]); recs['hybrid'] = H.predict(clf, feats, Xv, fv, fb)
hrs = fv.hist.groupby('user_id').sec.sum().reindex(fv.users).fillna(0).values / 3600
terc = pd.qcut(hrs, 3, labels=['light', 'medium', 'heavy'])
seen = fv.hist.groupby('user_id').tv_show_id.agg(set).to_dict()
rows = []
for m, rr in recs.items():
    ap = np.array([apk(fv.truth.get(u, set()), r) for u, r in zip(fv.users, rr)])
    fam = np.mean([np.mean([x in seen.get(u, ()) for x in r]) for u, r in zip(fv.users, rr)])
    hit_new = sum(x in fv.truth.get(u, ()) and x not in seen.get(u, ()) for u, r in zip(fv.users, rr) for x in r)
    hit_all = sum(x in fv.truth.get(u, ()) for u, r in zip(fv.users, rr) for x in r)
    rows.append({'method': m, **{t: round(ap[terc == t].mean(), 4) for t in ['light', 'medium', 'heavy']},
                 'share_already_watched': round(fam, 3), 'hits_on_new_programmes': f'{hit_new}/{hit_all}',
                 'distinct_ids_in_top5': len({x for r in rr for x in r})})
print('VAL (history wk11-24 -> wk25-30); terciles of history hours:', pd.Series(hrs).groupby(terc, observed=True).agg(['min', 'max']).round(1).to_dict('index'))
print(pd.DataFrame(rows).to_string(index=False))
t = pickle.load(open('../analysis/test_recs.pkl', 'rb'))
seen_t = g.groupby('user_id').tv_show_id.agg(set).to_dict()
cat = s[s.tv_show_id != 0].groupby('tv_show_id').tv_show_category.agg(lambda x: x.value_counts().index[0])
print('\nTEST Top-5 (weeks 31-42): share already watched in wk11-30, distinct ids, category mix')
for m, rr in t['recs'].items():
    fam = np.mean([np.mean([x in seen_t.get(u, ()) for x in r]) for u, r in zip(t['users'], rr)])
    cm = pd.Series([cat.get(x) for r in rr for x in r]).value_counts(normalize=True).round(2).head(4).to_dict()
    print(f'{m:12s} watched-before={fam:.3f} distinct={len({x for r in rr for x in r})} {cm}')
