"""Interaction-weight choice for the user-history signal (val, official target). Output: analysis/weights_check.txt"""
import numpy as np, scipy.sparse as sp
from core import *
import methods as M
g, s, users = load(); f = Fold('val', g, s, users); fb = M.pop_order(f)
h = f.hist[f.hist.user_id.isin(f.uidx) & f.hist.tv_show_id.isin(f.iidx)]
age = (f.h_end - h.air_t0.values) / (7 * 86400)
r = h.user_id.map(f.uidx).values; c = h.tv_show_id.map(f.iidx).values
for hl in [4, 8]:
    for name, w in [('watched hours', h.sec.values / 3600), ('# watched airings', np.ones(len(h))),
                    ('# airings with screen time >= 0.8', (h.frac.values >= .8) * 1.0), ('sum of screen-time fractions', h.frac.values)]:
        W = sp.csr_matrix((w * 0.5 ** (age / hl), (r, c)), shape=(len(users), len(f.items))).toarray()
        print(f'half-life {hl} wk | {name:34s} | MAP@5 = {mapk(f, topk(W, f.items, fb, f.cold)):.4f}')
