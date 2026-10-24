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
    'fit3':  ((11, 16), (17, 22)),   # extra sliding fitting folds (targets end <= wk 24, before the hold-out)
    'fit4':  ((11, 17), (18, 23)),
    'fit5':  ((11, 14), (15, 20)),
    # same history ends, 3-week targets (experiment: does the label-window length matter?)
    'w3a':   ((11, 18), (19, 21)), 'w3b': ((11, 15), (16, 18)), 'w3c': ((11, 16), (17, 19)),
    'w3d':   ((11, 17), (18, 20)), 'w3e': ((11, 14), (15, 17)),
    # 12-week targets (same length as the Kaggle test window), usable only for the final test model
    'L1':    ((11, 18), (19, 30)), 'L2': ((11, 16), (17, 28)), 'L3': ((11, 14), (15, 26)),
    'L4':    ((11, 17), (18, 29)), 'L5': ((11, 15), (16, 27)),
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
        self.truth = official_truth(g, t0, t1, users) if name != 'test' else None
        if name != 'test':   # per (user, programme) number of airings with screen time >= 0.8 in the target window
            tt = g[(g.wk >= t0) & (g.wk <= t1) & (g.frac >= 0.8)]
            self.target_counts = tt.groupby(['user_id', 'tv_show_id']).size()


def official_truth(g, w0, w1, users):
    """Target as documented on the competition's Data page: keep airings with screen time
    (watched / airing duration) >= 0.8, count them per user and programme, take the 5 most frequent.
    Ties: more watched seconds first, then smaller id (the organisers' tie rule is not documented)."""
    t = g[(g.wk >= w0) & (g.wk <= w1) & (g.frac >= 0.8) & g.user_id.isin(users)]
    c = t.groupby(['user_id', 'tv_show_id']).agg(n=('sec', 'size'), sec=('sec', 'sum')).reset_index()
    c = c.sort_values(['user_id', 'n', 'sec', 'tv_show_id'], ascending=[True, False, False, True])
    return c.groupby('user_id').head(5).groupby('user_id').tv_show_id.agg(set).to_dict()


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
