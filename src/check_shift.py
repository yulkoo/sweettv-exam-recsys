import numpy as np, pandas as pd
from data import *
v = load_views(); s = load_all_schedules()
v = v.sample(300000, random_state=0)
for h in [-3, -2, -1, 0, 1, 2, 3]:
    j = join(v, s, shift=h*3600)
    # how often does a view start within 2 min of an airing start (zapping to a programme start)
    jj = j.merge(v[['t0']], left_on='vidx', right_index=True).merge(s[['t0']].rename(columns={'t0':'s0'}), left_on='sidx', right_index=True)
    near = (np.abs(jj.t0 - (jj.s0 + h*3600)) < 120).mean()
    cov = j.vidx.nunique()/len(v)
    print(h, 'views matched', round(cov,3), 'start-near-airing-start', round(near,4))
