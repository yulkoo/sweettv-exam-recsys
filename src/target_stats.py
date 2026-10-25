"""Facts about the val target quoted in the report (output: analysis/target_stats.txt)."""
import numpy as np
from core import *
g, s, users = load()
f = Fold('val', g, s, users)
sizes = np.array([len(f.truth.get(u, ())) for u in users])
print('users with non-empty val target:', int((sizes > 0).sum()), 'of', len(users))
print('users with all five target programmes:', int((sizes == 5).sum()))
seen = f.hist.groupby('user_id').tv_show_id.agg(set)
items = [(u, i) for u in users for i in f.truth.get(u, ())]
new = sum(i not in seen.get(u, set()) for u, i in items)
print(f'val truth items never watched by the user in history wk 11-24: {new} of {len(items)} = {new / len(items):.3f}')
