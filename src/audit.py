"""Independent checks of the pipeline (output saved to analysis/audit.txt)."""
import numpy as np, pandas as pd
from core import *
import methods as M, hybrid as H

g, s, users, folds, data, clf, feats, recs = H.val_setup()
folds['test'] = Fold('test', g, s, users)

print('== 1. temporal separation')
for n, f in folds.items():
    tw = FOLDS[n][1]
    print(f'{n}: history airings {f.hist.air_t0.min()}..{f.hist.air_t0.max()} < target start {week_start(tw[0])}:',
          f.hist.air_t0.max() < week_start(tw[0]))
raw = pd.read_parquet(f'{D}/user_airing.parquet')
print('view log has any airing in weeks 31-42 (test):', bool((raw.air_t0 >= week_start(31)).any()))

print('== 2. individual methods + hybrid on val (official target: top-5 by # airings with screen time >= 0.8)')
fv = folds['val']; fb = M.pop_order(fv)
ap = {m: np.array([apk(fv.truth.get(u, set()), r) for u, r in zip(fv.users, rr)]) for m, rr in recs.items()}
for m, a in ap.items():
    print(f'{m:12s} MAP@5={a.mean():.4f}')

print('== 3. paired bootstrap over the 1,260 users (1000 resamples, seed 0): hybrid - other')
rng = np.random.default_rng(0)
B = rng.integers(0, len(users), (1000, len(users)))
for m in H.SCORERS:
    d = ap['hybrid'] - ap[m]
    bs = d[B].mean(1)
    print(f'hybrid - {m:12s} = {d.mean():+.4f}  95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]')

print('== 4. label-shuffle control: re-ranker fitted on permuted labels must lose its gain')
rng1 = np.random.default_rng(1)
Xs = [d[0].assign(cnt=rng1.permutation(d[0].cnt.values)) for d in (data['fit'], data['fit2'])]
clf_s, _ = H.fit(Xs)
print('hybrid with shuffled labels MAP@5 =', round(mapk(fv, H.predict(clf_s, feats, data['val'][0], fv, fb)), 4))

print('== 5. cold-start users get exactly the popularity Top-5 in every method')
pop5 = fb[:5]
for m, rr in recs.items():
    print(m, all(rr[i] == pop5 for i in np.where(fv.cold)[0]), 'n_cold =', int(fv.cold.sum()))

print('== 6. hand check of AP@5 for one user')
i = next(i for i, r in enumerate(recs['hybrid']) if 0 < sum(x in fv.truth.get(fv.users[i], ()) for x in r) < 5)
u = fv.users[i]; r = recs['hybrid'][i]; t = fv.truth.get(u, set())
hits = [x in t for x in r]
print('user', u, 'recs', r, 'hits', hits, '|truth| =', len(t))
prec = [sum(hits[:i + 1]) / (i + 1) for i in range(5)]
print('AP@5 by hand = sum(P@i * rel_i)/min(|truth|,5) =', sum(p for p, h in zip(prec, hits) if h) / min(len(t), 5),
      ' apk() =', apk(t, r))

print('== 7. LightGBM gain importance of the val-selection re-ranker (top 12)')
imp = pd.Series(clf.booster_.feature_importance('gain'), feats)
print((imp / imp.sum()).sort_values(ascending=False).head(12).round(3).to_string())
