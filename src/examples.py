"""Section 7/8 examples: two users, each method's Top-5 on the val hold-out (so they can be compared with what
the users actually watched in weeks 25-30), plus the final hybrid's Top-5 for the test weeks 31-42."""
import pickle, json, numpy as np, pandas as pd
from core import *
import methods as M, hybrid as H

g, s, users, folds, data, clf, feats, recs = H.val_setup()
fv = folds['val']; fb = M.pop_order(fv)
scv = data['val'][1]
fb = M.pop_order(fv)

# pick: heavy viewer = 95th percentile of history hours, light viewer = 10th percentile (both with val views)
hrs = fv.hist[fv.hist.user_id.isin(users)].groupby('user_id').sec.sum() / 3600
hrs = hrs[hrs.index.isin(fv.truth.keys())].sort_values()
heavy = hrs.index[int(.95 * len(hrs))]; light = hrs.index[int(.10 * len(hrs))]

# titles differ per airing (episode numbers, 'Спецпроект'); show the most frequent title of each id
lab_ = s[s.tv_show_id != 0]
meta = pd.DataFrame({'tv_show_title': lab_.groupby('tv_show_id').tv_show_title.agg(lambda x: x.value_counts().index[0]),
                     'tv_show_category': lab_.groupby('tv_show_id').tv_show_category.agg(lambda x: x.value_counts().index[0])})
def name(i): return f"{i} | {meta.tv_show_title.get(i, '?')[:45]} | {meta.tv_show_category.get(i, '?')}"

test = pickle.load(open('../analysis/test_recs.pkl', 'rb'))
out = {}
for lab, u in [('heavy', heavy), ('light', light)]:
    ui = fv.uidx[u]; truth = fv.truth[u]
    h = fv.hist[fv.hist.user_id == u]
    top_hist = h.groupby('tv_show_id').sec.sum().sort_values(ascending=False)
    print(f'\n##### {lab} viewer {u}: {hrs[u]:.1f} h labelled history (wk11-24), {h.tv_show_id.nunique()} programmes, '
          f'official top-{len(truth)} in wk25-30 (# airings with screen time >= 0.8)')
    print('history top-5 by hours:'); [print('   ', name(i), f'{v/3600:.1f} h') for i, v in top_hist.head(5).items()]
    res = {}
    for m, rr in recs.items():
        r = [int(x) for x in rr[ui]]
        ap = apk(truth, r)
        print(f'{m} AP@5={ap:.2f}')
        for x in r:
            print('   ', '*' if x in truth else ' ', 'seen' if x in set(top_hist.index) else 'new ', name(x))
        res[m] = {'ids': r, 'ap5': ap, 'hits': [x in truth for x in r], 'seen': [x in set(top_hist.index) for x in r]}
    tr = [int(x) for x in test['recs']['hybrid'][list(test['users']).index(u)]]
    print('final hybrid Top-5 for test wk31-42:'); [print('   ', name(x)) for x in tr]
    out[lab] = {'user': int(u), 'hours': float(hrs[u]), 'n_hist_items': int(h.tv_show_id.nunique()), 'n_truth': len(truth),
                'history_top': [int(i) for i in top_hist.head(5).index], 'val': res, 'test_hybrid': tr}
json.dump(out, open('../analysis/examples.json', 'w'), ensure_ascii=False, indent=1, default=str)
