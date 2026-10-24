"""Join the full view log to the full schedule and cache interaction rows."""
import time, numpy as np, pandas as pd
from data import *
t = time.time()
v = load_views(); s = load_all_schedules()
j = join(v, s)
j = j.merge(v[['user_id', 't0', 't1']], left_on='vidx', right_index=True)
j = j.merge(s[['tv_show_id', 'duration', 't0', 'channel_id']].rename(columns={'t0': 'air_t0', 'duration': 'air_dur'}), left_on='sidx', right_index=True)
print('pairs', len(j), 'unlabelled share of overlap seconds', round(j.loc[j.tv_show_id == 0, 'overlap'].sum() / j.overlap.sum(), 3))
j = j[j.tv_show_id != 0]
# aggregate several view fragments of the same airing (zapping back) into one user-airing record
g = j.groupby(['user_id', 'sidx'], as_index=False).agg(tv_show_id=('tv_show_id', 'first'), air_t0=('air_t0', 'first'),
                                                     air_dur=('air_dur', 'first'), channel_id=('channel_id', 'first'), sec=('overlap', 'sum'))
g['sec'] = np.minimum(g.sec, g.air_dur)
g['frac'] = g.sec / g.air_dur
g.to_parquet(D + '/user_airing.parquet')
s.to_parquet(D + '/schedule.parquet')
print('user-airing rows', len(g), 'users', g.user_id.nunique(), 'shows', g.tv_show_id.nunique(), 'sec', round(time.time() - t, 1))
print(g[['sec', 'frac', 'air_dur']].describe(percentiles=[.1, .25, .5, .75, .9]).to_string())
