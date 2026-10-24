"""Shared protocol: folds, 'watched' rule, candidate pool, MAP@5, top-5 extraction."""
import numpy as np, pandas as pd, os

D = os.path.join(os.path.dirname(__file__), '..', 'data')
K = 5
WEEK0 = pd.Timestamp('2020-03-09').value // 10**9      # Monday of ISO week 11, 2020
WEEK = 7 * 86400


def week_start(w):
    return WEEK0 + (w - 11) * WEEK


# history weeks (inclusive) -> target weeks (inclusive)
FOLDS = {
    'fit':   ((11, 18), (19, 24)),   # used only to fit the hybrid's fusion weights
    'fit2':  ((11, 15), (16, 21)),   # second, earlier fitting fold (more training rows for the re-ranker)
    'val':   ((11, 24), (25, 30)),   # the single documented hold-out for model selection
    'test':  ((11, 30), (31, 42)),   # final Kaggle prediction (no target known)
}


def watched(df):
    """A user 'watched' an airing if the overlap is >= min(300 s, 50% of the airing length)."""
    return df.sec >= np.minimum(300, 0.5 * df.air_dur)


def load():
    g = pd.read_parquet(f'{D}/user_airing.parquet')
    s = pd.read_parquet(f'{D}/schedule.parquet')
    g = g[watched(g)].copy()
    g['wk'] = 11 + (g.air_t0 - WEEK0) // WEEK
    s['wk'] = 11 + (s.t0 - WEEK0) // WEEK
    users = pd.read_csv(f'{D}/submission.csv').user_id.values
    return g, s, users


class Fold:
    def __init__(self, name, g, s, users):
        (h0, h1), (t0, t1) = FOLDS[name]
        self.name = name
        self.users = users
        self.hist = g[(g.wk >= h0) & (g.wk <= h1)]
        # candidate pool: every labelled programme that airs in the target window (schedule only)
        st = s[(s.wk >= t0) & (s.wk <= t1) & (s.tv_show_id != 0)]
        self.target_sched = st
        self.items = np.sort(st.tv_show_id.unique())
        self.hist_sched = s[(s.wk >= h0) & (s.wk <= h1)]
        self.h_end = week_start(h1 + 1)
        self.uidx = {u: i for i, u in enumerate(users)}
        self.iidx = {it: i for i, it in enumerate(self.items)}
        # cold-start: no labelled watched airing in the history window
        self.cold = ~np.isin(users, self.hist.user_id.unique())
        if name != 'test':
            tg = g[(g.wk >= t0) & (g.wk <= t1) & g.user_id.isin(users)]
            self.truth = tg.groupby('user_id').tv_show_id.agg(lambda x: set(x)).to_dict()
        else:
            self.truth = None


def topk(scores, items, fallback, cold=None, k=K):
    """scores: (n_users, n_items); <= -1e17 = not a candidate.
    Ties (e.g. all-zero rows, or fewer than 5 positive scores) are broken by the popularity order `fallback`,
    users in `cold` (no usable history) get the popularity Top-5 -- the same rule for every method."""
    prank = {it: r for r, it in enumerate(fallback)}
    tie = np.array([prank.get(it, len(fallback)) for it in items])
    out = []
    for u, row in enumerate(scores):
        if cold is not None and cold[u]:
            out.append(list(fallback[:k])); continue
        idx = np.lexsort((tie, -row))[:k]                  # primary: score desc, secondary: popularity
        rec = [items[i] for i in idx if row[i] > -1e17]
        if len(rec) < k:
            rec += [x for x in fallback if x not in rec][:k - len(rec)]
        out.append(rec[:k])
    return out


def apk(actual, pred, k=K):
    if not actual:
        return 0.0
    hits, s = 0, 0.0
    for i, p in enumerate(pred[:k]):
        if p in actual and p not in pred[:i]:
            hits += 1
            s += hits / (i + 1)
    return s / min(len(actual), k)


def mapk(fold, recs, k=K):
    """Mean over ALL test users; a user with no target views contributes 0."""
    return float(np.mean([apk(fold.truth.get(u, set()), r, k) for u, r in zip(fold.users, recs)]))
