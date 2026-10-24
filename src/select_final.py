"""Second selection round for the re-ranker (output: analysis/select_final.txt).
Every variant is scored on two hold-outs: 'val' (trained on the five 6-week folds whose targets end <= wk 24)
and 'fit' (wk 19-24, trained on fit5 only). A change is kept only if it helps on both."""
import os, pickle, sys, numpy as np, pandas as pd, lightgbm as lgb
from core import *
import hybrid as H, methods as M
C = f'{D}/cache'; os.makedirs(C, exist_ok=True)
g, s, users = load()
NAMES = H.FIT6 + ['w3a', 'w3b', 'w3c', 'w3d', 'w3e', 'val']
H.OWN = True
for n in NAMES:
    if not os.path.exists(f'{C}/{n}.pkl'):
        X, _ = H.build(Fold(n, g, s, users)); pickle.dump(X, open(f'{C}/{n}.pkl', 'wb'))
D_ = {n: pickle.load(open(f'{C}/{n}.pkl', 'rb')) for n in NAMES}
OWNC = [c for c in D_['val'].columns if c.startswith('own_')]
F = {n: Fold(n, g, s, users) for n in ['fit', 'val']}
OLD = dict(num_leaves=63, min_child_samples=100)          # first-round setting


def run(train, ev, seeds=(42,), own=False, **kw):
    drop = [] if own else OWNC
    Xe = D_[ev].drop(columns=drop); fv = F[ev]; fb = M.pop_order(fv); p = 0
    for sd in seeds:
        m, feats = H.fit([D_[n].drop(columns=drop) for n in train], seed=sd, **kw); p = p + m.predict(Xe[feats])
    S = np.full((len(fv.users), len(fv.items)), -1e18); S[Xe.user.values, Xe.item.values] = p
    return mapk(fv, topk(S, fv.items, fb, fv.cold))


def both(name, **kw):
    print(f'{name:58s} val {run(H.FIT6, "val", **kw):.4f}   fit {run(["fit5"], "fit", **kw):.4f}', flush=True)


print('== A. first-round re-ranker, 2 vs 5 training folds (val only)')
print(f'{"old setting, fit+fit2":58s} val {run(["fit", "fit2"], "val", **OLD):.4f}')
both('old setting, 5 folds', **OLD)
print('== B. personal (channel x hour) habit features for each programme')
both('old setting + own-slot features', own=True, **OLD)
print('== C. LightGBM LambdaRank settings (5 folds, seed 42)')
for kw in [dict(num_leaves=15), dict(num_leaves=31, min_child_samples=300), dict(num_leaves=15, n_estimators=1000),
           dict(OLD, n_estimators=250), dict(OLD, lambdarank_truncation_level=10), dict(OLD, objective='rank_xendcg'),
           dict(num_leaves=31, reg_lambda=10, colsample_bytree=0.5), dict(learning_rate=0.01, n_estimators=1500, num_leaves=31)]:
    both(str({k: v for k, v in kw.items() if OLD.get(k) != v} or kw), **kw)
print('== D. seed bag (5 seeds)')
both('num_leaves=31, min_child_samples=300, 5 seeds  (FINAL)', seeds=H.SEEDS)
print('== E. label-window length (val target = 6 weeks; same history ends, final setting)')
print(f'{"6-week labels (matches val window)":58s} val {run(H.FIT6, "val"):.4f}')
print(f'{"3-week labels":58s} val {run(["w3a", "w3b", "w3c", "w3d", "w3e"], "val"):.4f}')
print(f'{"6-week + 3-week labels mixed":58s} val {run(H.FIT6 + ["w3a", "w3b", "w3c", "w3d", "w3e"], "val"):.4f}')
print('== F. ensemble of LambdaRank with pointwise LightGBM models (per-user rank average)')
for ev, tr in [('val', H.FIT6), ('fit', ['fit5'])]:
    Xe = D_[ev].drop(columns=OWNC); fv = F[ev]; fb = M.pop_order(fv); u = Xe.user.values
    P = {'rank': 0}
    for sd in H.SEEDS:
        m, feats = H.fit([D_[n].drop(columns=OWNC) for n in tr], seed=sd); P['rank'] = P['rank'] + m.predict(Xe[feats])
    X = pd.concat([D_[n] for n in tr])
    kw = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=300, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, random_state=42, verbose=-1)
    P['poisson'] = lgb.LGBMRegressor(objective='poisson', **kw).fit(X[feats], np.minimum(X.cnt, 30), categorical_feature=['i_cat']).predict(Xe[feats])
    P['binary'] = lgb.LGBMClassifier(**kw).fit(X[feats], X.y.astype(int), categorical_feature=['i_cat']).predict_proba(Xe[feats])[:, 1]
    R = {k: pd.Series(v).groupby(u).rank(pct=True).values for k, v in P.items()}
    def sc(p):
        S = np.full((len(fv.users), len(fv.items)), -1e18); S[u, Xe.item.values] = p
        return round(mapk(fv, topk(S, fv.items, fb, fv.cold)), 4)
    print(ev, {'rank': sc(P['rank']), 'poisson': sc(P['poisson']), 'binary': sc(P['binary']),
               'rank+poisson': sc(R['rank'] + R['poisson']), 'rank+binary': sc(R['rank'] + R['binary']),
               'all three': sc(R['rank'] + R['poisson'] + R['binary'])}, flush=True)
