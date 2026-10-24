import sys, time, json, numpy as np
from core import *
import methods as M

g, s, users = load()
fold = Fold(sys.argv[1] if len(sys.argv) > 1 else 'val', g, s, users)
print(fold.name, 'candidates', len(fold.items), 'users with target', len(fold.truth))
fb = M.pop_order(fold)
res = {}
for name in sys.argv[2:] or ['popularity', 'history', 'cbf', 'assoc_rules', 'als', 'ppr']:
    t = time.time()
    sc = getattr(M, name)(fold)
    recs = topk(sc, fold.items, fb, fold.cold)
    res[name] = round(mapk(fold, recs), 5)
    print(f'{name:12s} MAP@5={res[name]:.5f}  {time.time() - t:.1f}s', flush=True)
