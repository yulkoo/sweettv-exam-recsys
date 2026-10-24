"""Scorers. Each returns a dense (n_users, n_items) score matrix over fold.items."""
import numpy as np, pandas as pd, scipy.sparse as sp

HALF_LIFE_WEEKS = 4.0


def ui_matrix(fold, weight='time_decay'):
    """User x programme matrix over the candidate pool built from history only."""
    h = fold.hist[fold.hist.user_id.isin(fold.uidx) & fold.hist.tv_show_id.isin(fold.iidx)]
    age_w = (fold.h_end - h.air_t0) / (7 * 86400)
    if weight == 'count':
        w = np.ones(len(h))
    elif weight == 'sec':
        w = h.sec.values / 3600
    else:   # recency-decayed watched hours
        w = (h.sec.values / 3600) * 0.5 ** (age_w.values / HALF_LIFE_WEEKS)
    r = h.user_id.map(fold.uidx).values; c = h.tv_show_id.map(fold.iidx).values
    return sp.csr_matrix((w, (r, c)), shape=(len(fold.users), len(fold.items)))


def all_item_matrix(fold):
    """User x programme over ALL history items (incl. ones not airing in target), all users (for co-view)."""
    h = fold.hist
    users = np.sort(h.user_id.unique()); items = np.sort(h.tv_show_id.unique())
    age_w = (fold.h_end - h.air_t0) / (7 * 86400)
    w = (h.sec.values / 3600) * 0.5 ** (age_w.values / HALF_LIFE_WEEKS)
    r = np.searchsorted(users, h.user_id.values); c = np.searchsorted(items, h.tv_show_id.values)
    return sp.csr_matrix((w, (r, c)), shape=(len(users), len(items))), users, items


def target_airtime(fold):
    """Hours each candidate programme is on air in the target window (schedule only, no views)."""
    a = fold.target_sched.groupby('tv_show_id').duration.sum() / 3600
    return a.reindex(fold.items).values


# ---------------- popularity baseline ----------------
def popularity(fold):
    h = fold.hist[fold.hist.tv_show_id.isin(fold.iidx)]
    recent = h[h.air_t0 >= fold.h_end - 4 * 7 * 86400]
    pop = recent.groupby('tv_show_id').user_id.nunique().reindex(fold.items).fillna(0).values
    return np.tile(pop.astype(float), (len(fold.users), 1))


def pop_order(fold):
    return list(fold.items[np.argsort(-popularity(fold)[0], kind='stable')])


# ---------------- 1. content-based filtering ----------------
def item_features(fold):
    """Bag of attributes per programme: category, genres, director(s), actors, decade."""
    s = pd.concat([fold.hist_sched, fold.target_sched])
    s = s[s.tv_show_id != 0].drop_duplicates('tv_show_id', keep='last').set_index('tv_show_id')
    feats = {}
    for it, row in s.iterrows():
        f = []
        if isinstance(row.tv_show_category, str): f.append('cat=' + row.tv_show_category)
        for gcol in ['tv_show_genre_1', 'tv_show_genre_2', 'tv_show_genre_3']:
            if isinstance(row[gcol], str): f.append('gen=' + row[gcol].strip().lower())
        if isinstance(row.director, str):
            f += ['dir=' + x.strip() for x in row.director.split(',') if x.strip()]
        if isinstance(row.actors, str):
            f += ['act=' + x.strip() for x in row.actors.split(',')[:10] if x.strip()]
        y = str(row.year_of_production)[:4]
        if y.isdigit(): f.append('dec=' + y[:3])
        f.append('ch=' + str(row.channel_id))
        feats[it] = f
    return feats


def cbf(fold):
    from sklearn.feature_extraction.text import TfidfVectorizer
    feats = item_features(fold)
    X_all, users, hist_items = all_item_matrix(fold)
    all_items = np.union1d(hist_items, fold.items)
    vec = TfidfVectorizer(analyzer=lambda x: x, min_df=2, sublinear_tf=True)
    F = vec.fit_transform([feats.get(i, []) for i in all_items])     # rows L2-normalised
    pos_h = np.searchsorted(all_items, hist_items); pos_c = np.searchsorted(all_items, fold.items)
    U = ui_matrix_hist_items(fold, users, hist_items, X_all)          # target users x hist items
    prof = U @ F[pos_h]                                                # weighted sum of item vectors
    prof = sp.csr_matrix(prof.multiply(1 / np.maximum(sp.linalg.norm(prof, axis=1), 1e-9)[:, None]))
    return np.asarray((prof @ F[pos_c].T).todense())


def ui_matrix_hist_items(fold, users, hist_items, X_all):
    rows = np.array([np.searchsorted(users, u) if u in set(users) else -1 for u in fold.users])
    out = sp.lil_matrix((len(fold.users), X_all.shape[1]))
    ok = rows >= 0
    out[np.where(ok)[0]] = X_all[rows[ok]]
    return out.tocsr()


# ---------------- 2. association rules ----------------
def assoc_rules(fold, min_support=20, min_conf=0.05):
    """Transactions = user-week baskets. Rules A->B scored by confidence * lift-filter; user score = sum over antecedents."""
    h = fold.hist
    b = h.groupby(['user_id', 'wk']).tv_show_id.unique()
    items = np.sort(h.tv_show_id.unique()); ii = {x: i for i, x in enumerate(items)}
    rows = np.repeat(np.arange(len(b)), b.str.len().values)
    cols = np.array([ii[x] for arr in b.values for x in arr])
    B = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(b), len(items)))
    n = B.shape[0]
    sup = np.asarray(B.sum(0)).ravel()
    C = (B.T @ B).tocoo()                        # pair supports
    m = (C.data >= min_support) & (C.row != C.col)
    a, c, sab = C.row[m], C.col[m], C.data[m]
    conf = sab / sup[a]
    lift = conf / (sup[c] / n)
    keep = (conf >= min_conf) & (lift > 1)
    R = sp.csr_matrix((conf[keep], (a[keep], c[keep])), shape=(len(items), len(items)))
    # antecedents = programmes in the user's last 4 history weeks (a 'recent basket')
    last = h[h.wk > FOLD_LAST(fold) - 4]
    U = sp.lil_matrix((len(fold.users), len(items)))
    for u, grp in last.groupby('user_id'):
        if u in fold.uidx:
            U[fold.uidx[u], [ii[x] for x in grp.tv_show_id.unique()]] = 1
    S = (U.tocsr() @ R).toarray()
    pos = np.searchsorted(items, fold.items); ok = (pos < len(items)) & (items[np.minimum(pos, len(items) - 1)] == fold.items)
    out = np.zeros((len(fold.users), len(fold.items)))
    out[:, ok] = S[:, pos[ok]]
    return out


def FOLD_LAST(fold):
    return int(fold.hist.wk.max())


# ---------------- 3. matrix factorisation (implicit ALS) ----------------
def als(fold, factors=64, reg=0.05, alpha=10.0, iters=20, seed=42):
    from implicit.als import AlternatingLeastSquares
    X, users, items = all_item_matrix(fold)
    Xc = X.copy(); Xc.data = np.log1p(alpha * Xc.data)            # confidence = log(1 + alpha * decayed hours)
    m = AlternatingLeastSquares(factors=factors, regularization=reg, iterations=iters, random_state=seed)
    m.fit(Xc, show_progress=False)
    uf = m.user_factors; itf = m.item_factors
    uf = uf.to_numpy() if hasattr(uf, 'to_numpy') else np.asarray(uf)
    itf = itf.to_numpy() if hasattr(itf, 'to_numpy') else np.asarray(itf)
    out = np.zeros((len(fold.users), len(fold.items)))
    pu = np.searchsorted(users, fold.users); oku = (pu < len(users)) & (users[np.minimum(pu, len(users) - 1)] == fold.users)
    pi = np.searchsorted(items, fold.items); oki = (pi < len(items)) & (items[np.minimum(pi, len(items) - 1)] == fold.items)
    out[np.ix_(oku, oki)] = uf[pu[oku]] @ itf[pi[oki]].T
    out[:, ~oki] = -1.0                                             # never-watched programmes rank below known ones
    return out


# ---------------- 4. personalised PageRank on co-view graph ----------------
def ppr(fold, alpha=0.3, iters=20, top_nb=50):
    """Item-item co-view graph (weights = cosine of user vectors, top-50 neighbours), random walk with restart
    to the user's decayed watch-time distribution. alpha = restart probability."""
    X, users, items = all_item_matrix(fold)
    Xb = X.copy(); Xb.data = np.log1p(Xb.data)
    norms = np.sqrt(np.asarray(Xb.multiply(Xb).sum(0)).ravel()) + 1e-9
    Xn = Xb.multiply(1 / norms).tocsc()
    S = (Xn.T @ Xn).tocsr(); S.setdiag(0); S.eliminate_zeros()
    # sparsify to top neighbours
    rows, cols, vals = [], [], []
    for i in range(S.shape[0]):
        st, en = S.indptr[i], S.indptr[i + 1]
        if en > st:
            d, c = S.data[st:en], S.indices[st:en]
            o = np.argsort(-d)[:top_nb]
            rows += [i] * len(o); cols += list(c[o]); vals += list(d[o])
    W = sp.csr_matrix((vals, (rows, cols)), shape=S.shape)
    W = W + W.T
    P = sp.diags(1 / np.maximum(np.asarray(W.sum(1)).ravel(), 1e-12)) @ W     # row-stochastic transition
    pu = np.searchsorted(users, fold.users); oku = (pu < len(users)) & (users[np.minimum(pu, len(users) - 1)] == fold.users)
    R = sp.lil_matrix((len(fold.users), len(items)))
    R[np.where(oku)[0]] = X[pu[oku]]
    R = R.tocsr(); R = sp.diags(1 / np.maximum(np.asarray(R.sum(1)).ravel(), 1e-12)) @ R
    R = R.toarray(); pi_ = R.copy()
    for _ in range(iters):
        pi_ = alpha * R + (1 - alpha) * (pi_ @ P)
    pi = np.searchsorted(items, fold.items); oki = (pi < len(items)) & (items[np.minimum(pi, len(items) - 1)] == fold.items)
    out = np.zeros((len(fold.users), len(fold.items)))
    out[:, oki] = pi_[:, pi[oki]]
    return out


# ---------------- personal re-watch signal (used inside the hybrid) ----------------
def history(fold):
    return ui_matrix(fold, 'time_decay').toarray()
