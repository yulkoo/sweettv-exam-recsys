"""Model selection for the re-ranker on the val fold only."""
import numpy as np, pandas as pd, lightgbm as lgb, pickle
from core import *
import methods as M, hybrid as H
g, s, users = load()
folds = {n: Fold(n, g, s, users) for n in ['fit2', 'fit', 'val']}
data = {n: H.build(f)[0] for n, f in folds.items()}
fv = folds['val']; fb = M.pop_order(fv)
Xv = data['val']; feats = [c for c in Xv.columns if c not in ('user', 'item', 'y', '_f')]

def ev(p):
    S = np.full((len(fv.users), len(fv.items)), -1e18); S[Xv.user.values, Xv.item.values] = p
    return round(mapk(fv, topk(S, fv.items, fb, fv.cold)), 4)

def lgb_rank(Xs, seed=42):
    X = pd.concat(Xs).sort_values(['_f', 'user'], kind='stable')
    grp = X.groupby(['_f', 'user'], sort=False).size().values
    m = lgb.LGBMRanker(objective='lambdarank', n_estimators=500, learning_rate=0.03, num_leaves=63,
                       min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                       lambdarank_truncation_level=20, eval_at=[5], random_state=seed, verbose=-1)
    m.fit(X[feats], X.y, group=grp, categorical_feature=['i_cat'])
    return m

for n in data: data[n]['_f'] = n
res = {}
clf, _ = H.fit_pointwise([data['fit']]);                       res['HGB pointwise, fit'] = ev(clf.predict_proba(Xv[feats])[:, 1])
clf, _ = H.fit_pointwise([data['fit'], data['fit2']]);         res['HGB pointwise, fit+fit2'] = ev(clf.predict_proba(Xv[feats])[:, 1])
m = lgb_rank([data['fit']]);                         res['LGBM lambdarank, fit'] = ev(m.predict(Xv[feats]))
m = lgb_rank([data['fit'], data['fit2']]);           res['LGBM lambdarank, fit+fit2'] = ev(m.predict(Xv[feats]))
for k, v in res.items(): print(f'{k:28s} val MAP@5 = {v}')
