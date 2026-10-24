"""Re-ranker label choice under the official target (val only)."""
import numpy as np, pandas as pd, lightgbm as lgb
from core import *
import methods as M, hybrid as H
g, s, users = load()
folds = {n: Fold(n, g, s, users) for n in ['fit2', 'fit', 'val']}
data = {n: H.build(f)[0] for n, f in folds.items()}
fv = folds['val']; fb = M.pop_order(fv); Xv = data['val']
feats = [c for c in Xv.columns if c not in H.NONFEAT]

def target_counts(f):
    (_, _), (w0, w1) = FOLDS[f.name]
    t = g[(g.wk >= w0) & (g.wk <= w1) & (g.frac >= 0.8)]
    return t.groupby(['user_id', 'tv_show_id']).size()

def labels(X, f, kind):
    c = target_counts(f)
    key = pd.MultiIndex.from_arrays([f.users[X.user.values], f.items[X.item.values]])
    n = c.reindex(key).fillna(0).values
    if kind == 'top5': return X.y.astype(int).values
    if kind == 'count_cap30': return np.minimum(n, 30).astype(int)
    if kind == 'log2_count': return np.floor(np.log2(1 + n)).astype(int)
    if kind == 'top5_graded':  # 5..1 by truth rank, else 0; plus 0 for non-top
        return X.y.astype(int).values * (1 + np.minimum(np.floor(np.log2(1 + n)), 4)).astype(int)

def fit_eval(train, kind, seed=42):
    X = pd.concat([data[n].assign(_f=i, lab=labels(data[n], folds[n], kind)) for i, n in enumerate(train)]).sort_values(['_f', 'user'], kind='stable')
    grp = X.groupby(['_f', 'user'], sort=False).size().values
    m = lgb.LGBMRanker(objective='lambdarank', n_estimators=500, learning_rate=0.03, num_leaves=63, min_child_samples=100,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, lambdarank_truncation_level=20,
                       label_gain=list(range(64)), random_state=seed, verbose=-1)
    m.fit(X[feats], X.lab, group=grp, categorical_feature=['i_cat'])
    S = np.full((len(fv.users), len(fv.items)), -1e18); S[Xv.user.values, Xv.item.values] = m.predict(Xv[feats])
    return round(mapk(fv, topk(S, fv.items, fb, fv.cold)), 4)

print('candidate recall of val truth:', round(Xv.y.sum() / sum(len(v) for v in fv.truth.values()), 4))
for kind in ['top5', 'count_cap30', 'log2_count', 'top5_graded']:
    for train in [['fit'], ['fit', 'fit2']]:
        print(f'labels={kind:12s} train={"+".join(train):9s} val MAP@5 = {fit_eval(train, kind)}', flush=True)
