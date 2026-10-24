"""Hybrid: candidate generation from the individual methods + gradient-boosted re-ranking of their scores.
The re-ranker is fitted on fold 'fit' (history wk11-18 -> target wk19-24) and evaluated on 'val'
(history wk11-24 -> target wk25-30). For the Kaggle file it is refitted on fit+val and applied to 'test'."""
import sys, time, json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from core import *
import methods as M

N_CAND = 100          # per method, per user
SEED = 42
OWN = False          # personal (channel x hour) habit features: tried, did not help on either hold-out -> off
SEEDS = (42, 43, 44, 45, 46)   # final re-ranker = average of 5 LambdaRank models
FIT6 = ['fit', 'fit2', 'fit3', 'fit4', 'fit5']   # 6-week targets ending <= wk 24 (selection / val evaluation)
FIT12 = ['L1', 'L2', 'L3', 'L4', 'L5']             # 12-week targets (= test window length), final test model
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
    nweeks = ts.wk.nunique()            # 6 in fit/val, 12 in test -> all schedule features are per week
    Ic = np.zeros((ni, len(chans))); Icn = np.zeros((ni, len(chans)))
    a = ts.groupby(['tv_show_id', 'channel_id']).duration.agg(['sum', 'size'])
    for (it, c), v in a.iterrows():
        Ic[fold.iidx[it], ci[c]] = v['sum'] / 3600 / nweeks
        Icn[fold.iidx[it], ci[c]] = v['size'] / nweeks
    chan_air = Uc @ Ic.T
    chan_nair = Uc @ Icn.T              # expected airings per week on the user's channels
    # user hour-of-week profile vs item hour-of-week airing profile (168 slots)
    def how(t): return ((t - WEEK0) % WEEK) // 3600
    hu = h.assign(s=how(h.air_t0)).groupby(['user_id', 's']).sec.sum()
    hu = hu / hu.groupby(level=0).transform('sum')
    Uh = np.zeros((nu, 168))
    for (u, sl), v in hu.items(): Uh[fold.uidx[u], int(sl)] = v
    hi = ts.assign(s=how(ts.t0)).groupby(['tv_show_id', 's']).duration.sum() / 3600 / nweeks
    Ih = np.zeros((ni, 168))
    for (it, sl), v in hi.items(): Ih[fold.iidx[it], int(sl)] = v
    hour_air = Uh @ Ih.T
    # joint (channel, hour-of-week) habit x airings per week in that cell
    hc = h.assign(s=how(h.air_t0)).groupby(['user_id', 'channel_id', 's']).sec.sum()
    hc = hc / hc.groupby(level=0).transform('sum')
    tc = ts.assign(s=how(ts.t0)).groupby(['channel_id', 's', 'tv_show_id']).size() / nweeks
    j = hc.rename('w').reset_index().merge(tc.rename('n').reset_index(), on=['channel_id', 's'])
    j = j.assign(v=j.w * j.n).groupby(['user_id', 'tv_show_id']).v.sum()
    slot_nair = np.zeros((nu, ni))
    slot_nair[[fold.uidx[u] for u in j.index.get_level_values(0)], [fold.iidx[i] for i in j.index.get_level_values(1)]] = j.values
    return {'chan_air': chan_air, 'chan_nair': chan_nair, 'hour_air': hour_air, 'slot_nair': slot_nair}


def history_stats(fold):
    h = fold.hist[fold.hist.user_id.isin(fold.uidx) & fold.hist.tv_show_id.isin(fold.iidx)]
    last_wk = h.wk.max() if len(h) else 0
    hw = fold.hist.wk.max()
    h = h.assign(hi=(h.frac >= 0.8) * 1.0, hi4=((h.frac >= 0.8) & (h.wk > hw - 4)) * 1.0, hi2=((h.frac >= 0.8) & (h.wk > hw - 2)) * 1.0)
    KEYS = ['n_air', 'hours', 'n_wk', 'frac', 'n80', 'n80_4w', 'n80_2w', 'fsum']
    st = h.groupby(['user_id', 'tv_show_id']).agg(n_air=('sec', 'size'), hours=('sec', lambda x: x.sum() / 3600),
                                                 n_wk=('wk', 'nunique'), last=('wk', 'max'), frac=('frac', 'mean'),
                                                 n80=('hi', 'sum'), n80_4w=('hi4', 'sum'), n80_2w=('hi2', 'sum'), fsum=('frac', 'sum'))
    out = {k: np.zeros((len(fold.users), len(fold.items))) for k in KEYS}
    out['since'] = np.full((len(fold.users), len(fold.items)), 99.0)
    r = np.array([fold.uidx[u] for u in st.index.get_level_values(0)]); c = np.array([fold.iidx[i] for i in st.index.get_level_values(1)])
    for k in KEYS: out[k][r, c] = st[k].values
    out['since'][r, c] = fold.hist.wk.max() - st['last'].values
    # history-window weekly rates (history length differs between folds)
    nh = fold.hist.wk.nunique()
    for k in ['n_air', 'hours', 'n80', 'fsum']: out[k] = out[k] / nh
    out['n80_4w'] /= 4; out['n80_2w'] /= 2
    out['n80_rank'] = np.argsort(np.argsort(-out['n80'], axis=1, kind='stable'), axis=1)
    return out


def item_stats(fold):
    h = fold.hist[fold.hist.tv_show_id.isin(fold.iidx)]
    hw = fold.hist.wk.max()
    def pop(w0): return h[h.wk > hw - w0].groupby('tv_show_id').user_id.nunique().reindex(fold.items).fillna(0).values
    h80 = h[h.frac >= 0.8]
    def pop80(w0): return h80[h80.wk > hw - w0].groupby('tv_show_id').user_id.nunique().reindex(fold.items).fillna(0).values
    ts = fold.target_sched
    nweeks = ts.wk.nunique()
    air_h = (ts.groupby('tv_show_id').duration.sum() / 3600).reindex(fold.items).fillna(0).values / nweeks
    n_air = ts.groupby('tv_show_id').size().reindex(fold.items).fillna(0).values / nweeks
    hist_air = (fold.hist_sched[fold.hist_sched.tv_show_id != 0].groupby('tv_show_id').duration.sum() / 3600
                ).reindex(fold.items).fillna(0).values / fold.hist.wk.nunique()
    cat = ts.drop_duplicates('tv_show_id').set_index('tv_show_id').tv_show_category.reindex(fold.items)
    cat_code = pd.Categorical(cat, categories=sorted(ts.tv_show_category.dropna().unique())).codes
    mfrac = h.groupby('tv_show_id').frac.mean().reindex(fold.items).fillna(0).values
    return {'pop1': pop(1), 'pop2': pop(2), 'pop4': pop(4), 'popall': pop(99) / fold.hist.wk.nunique(),
            'pop80_1': pop80(1), 'pop80_4': pop80(4), 'mean_frac': mfrac, 'air_h_wk': air_h,
            'n_air_wk': n_air, 'air_ratio': air_h / (hist_air + 0.1), 'cat': cat_code.astype(float)}


def own_slot(fold):
    """Per (user, programme): where (channel x hour) the user watched THIS programme with screen time >= 0.8,
    share of its past airings in that cell they watched, times its airings per week in that cell in the target
    schedule = expected >= 0.8 views per week if the habit repeats. Hour-of-week and hour-of-day granularity,
    full history and last 4 history weeks."""
    nu, ni = len(fold.users), len(fold.items)
    ts = fold.target_sched; nweeks = ts.wk.nunique(); hw = fold.hist.wk.max()
    hs = fold.hist_sched[fold.hist_sched.tv_show_id.isin(fold.iidx)]
    h = fold.hist[fold.hist.user_id.isin(fold.uidx) & fold.hist.tv_show_id.isin(fold.iidx) & (fold.hist.frac >= 0.8)]
    out = {}
    for gran, cell in [('w', lambda t: ((t - WEEK0) % WEEK) // 3600), ('d', lambda t: ((t - WEEK0) % 86400) // 3600)]:
        tn = ts.assign(c=cell(ts.t0)).groupby(['tv_show_id', 'channel_id', 'c']).size().rename('tn') / nweeks
        for win in [99, 4]:
            hh = h[h.wk > hw - win]; ss = hs[hs.wk > hw - win]
            sn = ss.assign(c=cell(ss.t0)).groupby(['tv_show_id', 'channel_id', 'c']).size().rename('sn')
            u = hh.assign(c=cell(hh.air_t0)).groupby(['user_id', 'tv_show_id', 'channel_id', 'c']).size().rename('un').reset_index()
            u = u.join(sn, on=['tv_show_id', 'channel_id', 'c']).join(tn, on=['tv_show_id', 'channel_id', 'c'])
            u = u.assign(v=np.minimum(u.un / u.sn.clip(lower=1), 1) * u.tn.fillna(0), k=u.tn.fillna(0))
            a = u.groupby(['user_id', 'tv_show_id'])[['v', 'k']].sum()
            r = a.index.get_level_values(0).map(fold.uidx).values; c = a.index.get_level_values(1).map(fold.iidx).values
            for nm, col in [('exp', 'v'), ('nair', 'k')]:
                m = np.zeros((nu, ni)); m[r, c] = a[col].values
                out[f'own_{nm}_{gran}{win}'] = m
    return out


def build(fold, cache=None):
    t = time.time()
    sc = {m: getattr(M, m)(fold) for m in SCORERS}
    hs = history_stats(fold); ctx = user_item_context(fold); its = item_stats(fold)
    own = own_slot(fold) if OWN else {}
    nu, ni = len(fold.users), len(fold.items)
    # candidates: union of each method's top-N plus every history programme that airs again
    cand = np.zeros((nu, ni), bool)
    for m in SCORERS:
        top = np.argsort(-sc[m], axis=1)[:, :N_CAND]
        cand[np.arange(nu)[:, None], top] = True
    # schedule-driven candidates: what airs most often in the user's (channel, hour-of-week) slots / channels
    for k in ['slot_nair', 'chan_nair']:
        top = np.argsort(-ctx[k], axis=1)[:, :N_CAND]
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
    for k, v in own.items():
        X[k] = v[ui, ii]
        X[k + '_rank'] = np.argsort(np.argsort(-v, axis=1, kind='stable'), axis=1)[ui, ii]
    # direct estimate of the target quantity: per-airing completion rate in history x airings per week ahead
    hist_n = fold.hist_sched[fold.hist_sched.tv_show_id != 0].groupby('tv_show_id').size().reindex(fold.items).fillna(0).values
    nh = fold.hist.wk.nunique()
    rate80 = hs['n80'][ui, ii] * nh / np.maximum(hist_n[ii], 1)          # share of past airings watched >= 80%
    X['rate80'] = rate80
    X['exp80'] = rate80 * its['n_air_wk'][ii]
    X['exp80_slot'] = rate80 * ctx['slot_nair'][ui, ii]
    X['exp80_rank'] = pd.Series(X['exp80']).groupby(ui).rank(ascending=False, method='first').values
    uh = fold.hist[fold.hist.user_id.isin(fold.uidx)].groupby('user_id').agg(uh=('sec', 'sum'), un=('tv_show_id', 'nunique'))
    uh = uh.reindex(fold.users).fillna(0)
    X['u_hours'] = uh.uh.values[ui] / 3600; X['u_items'] = uh.un.values[ui]
    X = pd.DataFrame(X)
    X['user'] = ui; X['item'] = ii
    if fold.truth is not None:
        X['y'] = [fold.items[i] in fold.truth.get(fold.users[u], ()) for u, i in zip(ui, ii)]
        key = pd.MultiIndex.from_arrays([fold.users[ui], fold.items[ii]])
        X['cnt'] = fold.target_counts.reindex(key).fillna(0).values.astype(int)
    msg = f'{fold.name}: {len(X)} rows, {cand.sum(1).mean():.0f} cand/user, {time.time() - t:.0f}s'
    if fold.truth is not None:
        msg += f', positives {X.y.mean():.3f}'
    print(msg, flush=True)
    return X, sc


NONFEAT = ('user', 'item', 'y', '_f', 'cnt', 'lab')


def fit_pointwise(Xs):
    """Alternative re-ranker (kept for the comparison in select_hybrid.py)."""
    X = pd.concat(Xs)
    feats = [c for c in X.columns if c not in NONFEAT]
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=100,
                                         l2_regularization=1.0, categorical_features=[feats.index('i_cat')],
                                         random_state=SEED)
    clf.fit(X[feats], X.y)
    return clf, feats


def fit(Xs, seed=SEED, label='cnt', **kw):
    """Final re-ranker: LightGBM LambdaRank, one query group per user. Graded label = number of airings the user
    watched with screen time >= 0.8 in the target window (capped at 30) -- the quantity the official top-5 is ranked by."""
    import lightgbm as lgb
    X = pd.concat([x.assign(_f=i) for i, x in enumerate(Xs)]).sort_values(['_f', 'user'], kind='stable')
    feats = [c for c in X.columns if c not in NONFEAT]
    y = np.minimum(X[label].values, 30).astype(int)
    grp = X.groupby(['_f', 'user'], sort=False).size().values
    p = dict(objective='lambdarank', n_estimators=500, learning_rate=0.03, num_leaves=31,
             min_child_samples=300, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
             lambdarank_truncation_level=20, label_gain=list(range(64)), random_state=seed, verbose=-1)
    p.update(kw)
    m = lgb.LGBMRanker(**p)
    m.fit(X[feats], y, group=grp, categorical_feature=['i_cat'])
    return m, feats


def fit_bag(Xs):
    return [fit(Xs, seed=sd) for sd in SEEDS]


def predict(clf, feats, X, fold, fb):
    if isinstance(clf, list):        # seed bag
        p = sum(m.predict(X[f]) for m, f in clf)
    else:
        p = clf.predict_proba(X[feats])[:, 1] if hasattr(clf, 'predict_proba') else clf.predict(X[feats])
    S = np.full((len(fold.users), len(fold.items)), -1e18)
    S[X.user.values, X.item.values] = p
    return topk(S, fold.items, fb, fold.cold)


def val_setup():
    """Shared by audit/segments/examples: the selection-time hybrid (fitted on fit+fit2) and all methods on val."""
    g, s, users = load()
    folds = {n: Fold(n, g, s, users) for n in FIT6 + ['val']}
    data = {n: build(f) for n, f in folds.items()}
    for n in FIT6: data[n] = (data[n][0], None)          # free score matrices of training folds
    fv = folds['val']; fb = M.pop_order(fv)
    clf = fit_bag([data[n][0] for n in FIT6]); feats = None
    recs = {m: topk(data['val'][1][m], fv.items, fb, fv.cold) for m in SCORERS}
    recs['hybrid'] = predict(clf, feats, data['val'][0], fv, fb)
    return g, s, users, folds, data, clf, feats, recs


def write_sub(fold, recs, path):
    pd.DataFrame({'user_id': fold.users, 'tv_show_id': [' '.join(map(str, r)) for r in recs]}).to_csv(path, index=False)


if __name__ == '__main__':
    import pickle, time
    t0 = time.time()
    g, s, users = load()
    def frames(names):
        out = {}
        for n in names:
            X, _ = build(Fold(n, g, s, users)); out[n] = X.astype({c: 'float32' for c in X.columns if X[c].dtype == 'float64'})
        return out
    # 1) selection: 5 folds with 6-week targets ending <= wk 24 -> score on 'val' (target wk 25-30)
    tr6 = frames(FIT6)
    fv = Fold('val', g, s, users); Xv, scv = build(fv); fb = M.pop_order(fv)
    print('HYBRID val MAP@5', round(mapk(fv, predict(fit_bag(list(tr6.values())), None, Xv, fv, fb)), 5), flush=True)
    del scv
    ft = Fold('test', g, s, users); Xt, sct = build(ft); fbt = M.pop_order(ft)
    # 2a) final: same recipe, trained on five folds whose target window is 12 weeks like the test window
    tr12 = frames(FIT12)
    recs = {'hybrid': predict(fit_bag(list(tr12.values())), None, Xt, ft, fbt)}
    del tr12
    # 2b) alternative: 6-week-target folds incl. 'val' (submitted separately to compare on the public board)
    recs['hybrid_6w'] = predict(fit_bag(list(tr6.values()) + [Xv]), None, Xt, ft, fbt)
    for m in SCORERS:
        recs[m] = topk(sct[m], ft.items, fbt, ft.cold)
    write_sub(ft, recs['hybrid'], '../submissions/final_hybrid_lambdarank.csv')
    write_sub(ft, recs['hybrid_6w'], '../submissions/alt_hybrid_6week_labels.csv')
    write_sub(ft, recs['popularity'], '../submissions/popularity_baseline.csv')
    pickle.dump({'users': ft.users, 'recs': recs}, open('../analysis/test_recs.pkl', 'wb'))
    print('wrote submissions/final_hybrid_lambdarank.csv (+ alt_hybrid_6week_labels.csv), total runtime', round(time.time() - t0), 's')
