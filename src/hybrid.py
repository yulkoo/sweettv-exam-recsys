"""Hybrid: candidate generation from the individual methods + gradient-boosted re-ranking of their scores.
The re-ranker is fitted on fold 'fit' (history wk11-18 -> target wk19-24) and evaluated on 'val'
(history wk11-24 -> target wk25-30). For the Kaggle file it is refitted on fit+val and applied to 'test'."""
import sys, time, json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from core import *
import methods as M

N_CAND = 100          # per method, per user
SEED = 42
SCORERS = ['popularity', 'history', 'cbf', 'assoc_rules', 'als', 'ppr']


def user_item_context(fold):
    """Schedule-only features: how much of each candidate's target-window airtime falls on the user's
    channels and in the user's usual hours (user habits from history; airings from the target schedule)."""
    h = fold.hist[fold.hist.user_id.isin(fold.uidx)]
    nu, ni = len(fold.users), len(fold.items)
    # user channel shares
    ch = h.groupby(['user_id', 'channel_id']).sec.sum()
    ch = ch / ch.groupby(level=0).transform('sum')
    chans = np.sort(fold.target_sched.channel_id.unique()); ci = {c: i for i, c in enumerate(chans)}
    Uc = np.zeros((nu, len(chans)))
    for (u, c), v in ch.items():
        if c in ci: Uc[fold.uidx[u], ci[c]] = v
    ts = fold.target_sched
    Ic = np.zeros((ni, len(chans)))
    a = ts.groupby(['tv_show_id', 'channel_id']).duration.sum() / 3600
    for (it, c), v in a.items():
        Ic[fold.iidx[it], ci[c]] = v
    chan_air = Uc @ Ic.T
    # user hour-of-week profile vs item hour-of-week airing profile (168 slots)
    def how(t): return ((t - WEEK0) % WEEK) // 3600
    hu = h.assign(s=how(h.air_t0)).groupby(['user_id', 's']).sec.sum()
    hu = hu / hu.groupby(level=0).transform('sum')
    Uh = np.zeros((nu, 168))
    for (u, sl), v in hu.items(): Uh[fold.uidx[u], int(sl)] = v
    hi = ts.assign(s=how(ts.t0)).groupby(['tv_show_id', 's']).duration.sum() / 3600
    Ih = np.zeros((ni, 168))
    for (it, sl), v in hi.items(): Ih[fold.iidx[it], int(sl)] = v
    hour_air = Uh @ Ih.T
    # both: user's (channel, hour-of-week) habit
    return {'chan_air': chan_air, 'hour_air': hour_air}


def history_stats(fold):
    h = fold.hist[fold.hist.user_id.isin(fold.uidx) & fold.hist.tv_show_id.isin(fold.iidx)]
    last_wk = h.wk.max() if len(h) else 0
    st = h.groupby(['user_id', 'tv_show_id']).agg(n_air=('sec', 'size'), hours=('sec', lambda x: x.sum() / 3600),
                                                 n_wk=('wk', 'nunique'), last=('wk', 'max'), frac=('frac', 'mean'))
    out = {k: np.zeros((len(fold.users), len(fold.items))) for k in ['n_air', 'hours', 'n_wk', 'frac']}
    out['since'] = np.full((len(fold.users), len(fold.items)), 99.0)
    r = np.array([fold.uidx[u] for u in st.index.get_level_values(0)]); c = np.array([fold.iidx[i] for i in st.index.get_level_values(1)])
    for k in ['n_air', 'hours', 'n_wk', 'frac']: out[k][r, c] = st[k].values
    out['since'][r, c] = fold.hist.wk.max() - st['last'].values
    return out


def item_stats(fold):
    h = fold.hist[fold.hist.tv_show_id.isin(fold.iidx)]
    hw = fold.hist.wk.max()
    def pop(w0): return h[h.wk > hw - w0].groupby('tv_show_id').user_id.nunique().reindex(fold.items).fillna(0).values
    ts = fold.target_sched
    nweeks = ts.wk.nunique()
    air_h = (ts.groupby('tv_show_id').duration.sum() / 3600).reindex(fold.items).fillna(0).values / nweeks
    n_air = ts.groupby('tv_show_id').size().reindex(fold.items).fillna(0).values / nweeks
    hist_air = (fold.hist_sched[fold.hist_sched.tv_show_id != 0].groupby('tv_show_id').duration.sum() / 3600
                ).reindex(fold.items).fillna(0).values / fold.hist.wk.nunique()
    cat = ts.drop_duplicates('tv_show_id').set_index('tv_show_id').tv_show_category.reindex(fold.items)
    cat_code = pd.Categorical(cat, categories=sorted(ts.tv_show_category.dropna().unique())).codes
    return {'pop1': pop(1), 'pop2': pop(2), 'pop4': pop(4), 'popall': pop(99), 'air_h_wk': air_h,
            'n_air_wk': n_air, 'air_ratio': air_h / (hist_air + 0.1), 'cat': cat_code.astype(float)}


def build(fold, cache=None):
    t = time.time()
    sc = {m: getattr(M, m)(fold) for m in SCORERS}
    hs = history_stats(fold); ctx = user_item_context(fold); its = item_stats(fold)
    nu, ni = len(fold.users), len(fold.items)
    # candidates: union of each method's top-N plus every history programme that airs again
    cand = np.zeros((nu, ni), bool)
    for m in SCORERS:
        top = np.argsort(-sc[m], axis=1)[:, :N_CAND]
        cand[np.arange(nu)[:, None], top] = True
    cand |= hs['n_air'] > 0
    ui, ii = np.nonzero(cand)
    X = {}
    for m in SCORERS:
        X[m] = sc[m][ui, ii]
        rk = np.argsort(np.argsort(-sc[m], axis=1), axis=1)
        X[m + '_rank'] = rk[ui, ii]
    for k, v in hs.items(): X['h_' + k] = v[ui, ii]
    for k, v in ctx.items():
        X[k] = v[ui, ii]
        X[k + '_rank'] = np.argsort(np.argsort(-v, axis=1), axis=1)[ui, ii]
    for k, v in its.items(): X['i_' + k] = v[ii]
    uh = fold.hist[fold.hist.user_id.isin(fold.uidx)].groupby('user_id').agg(uh=('sec', 'sum'), un=('tv_show_id', 'nunique'))
    uh = uh.reindex(fold.users).fillna(0)
    X['u_hours'] = uh.uh.values[ui] / 3600; X['u_items'] = uh.un.values[ui]
    X = pd.DataFrame(X)
    X['user'] = ui; X['item'] = ii
    if fold.truth is not None:
        X['y'] = [fold.items[i] in fold.truth.get(fold.users[u], ()) for u, i in zip(ui, ii)]
    msg = f'{fold.name}: {len(X)} rows, {cand.sum(1).mean():.0f} cand/user, {time.time() - t:.0f}s'
    if fold.truth is not None:
        msg += f', positives {X.y.mean():.3f}'
    print(msg, flush=True)
    return X, sc


def fit_pointwise(Xs):
    """Alternative re-ranker (kept for the comparison in select_hybrid.py)."""
    X = pd.concat(Xs)
    feats = [c for c in X.columns if c not in ('user', 'item', 'y', '_f')]
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=100,
                                         l2_regularization=1.0, categorical_features=[feats.index('i_cat')],
                                         random_state=SEED)
    clf.fit(X[feats], X.y)
    return clf, feats


def fit(Xs, seed=SEED):
    """Final re-ranker: LightGBM LambdaRank, one query group per user."""
    import lightgbm as lgb
    X = pd.concat([x.assign(_f=i) for i, x in enumerate(Xs)]).sort_values(['_f', 'user'], kind='stable')
    feats = [c for c in X.columns if c not in ('user', 'item', 'y', '_f')]
    grp = X.groupby(['_f', 'user'], sort=False).size().values
    m = lgb.LGBMRanker(objective='lambdarank', n_estimators=500, learning_rate=0.03, num_leaves=63,
                       min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                       lambdarank_truncation_level=20, random_state=seed, verbose=-1)
    m.fit(X[feats], X.y, group=grp, categorical_feature=['i_cat'])
    return m, feats


def predict(clf, feats, X, fold, fb):
    p = clf.predict_proba(X[feats])[:, 1] if hasattr(clf, 'predict_proba') else clf.predict(X[feats])
    S = np.full((len(fold.users), len(fold.items)), -1e18)
    S[X.user.values, X.item.values] = p
    return topk(S, fold.items, fb, fold.cold)


def write_sub(fold, recs, path):
    pd.DataFrame({'user_id': fold.users, 'tv_show_id': [' '.join(map(str, r)) for r in recs]}).to_csv(path, index=False)


if __name__ == '__main__':
    import pickle, time
    t0 = time.time()
    g, s, users = load()
    folds = {n: Fold(n, g, s, users) for n in ['fit', 'val', 'test']}
    data = {n: build(f) for n, f in folds.items()}
    # 1) selection: fit on 'fit' (target wk19-24), score on 'val' (target wk25-30)
    clf, feats = fit([data['fit'][0]])
    fv = folds['val']; fb = M.pop_order(fv)
    print('HYBRID val MAP@5', round(mapk(fv, predict(clf, feats, data['val'][0], fv, fb)), 5))
    # 2) final: same recipe refitted on the most recent window ('val' fold), applied to 'test'
    clf2, _ = fit([data['val'][0]])
    ft = folds['test']; fbt = M.pop_order(ft)
    recs = {'hybrid': predict(clf2, feats, data['test'][0], ft, fbt)}
    for m in SCORERS:
        recs[m] = topk(data['test'][1][m], ft.items, fbt, ft.cold)
    write_sub(ft, recs['hybrid'], '../submissions/final_hybrid_lambdarank.csv')
    write_sub(ft, recs['popularity'], '../submissions/popularity_baseline.csv')
    pickle.dump({'users': ft.users, 'recs': recs}, open('../analysis/test_recs.pkl', 'wb'))
    print('wrote submissions/final_hybrid_lambdarank.csv, total runtime', round(time.time() - t0), 's')
