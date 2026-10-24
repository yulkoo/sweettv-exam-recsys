"""Loading and view->programme join (channel + interval overlap)."""
import numpy as np, pandas as pd, os

D = os.path.join(os.path.dirname(__file__), '..', 'data')
SCHED = ['11-20', '21-30', '31-42']


def load_views():
    v = pd.read_csv(f'{D}/dataset11-30.csv')
    v['t0'] = pd.to_datetime(v.start_time).astype('datetime64[s]').astype('int64')
    v['t1'] = pd.to_datetime(v.stop_time).astype('datetime64[s]').astype('int64')
    return v[['user_id', 'vsetv_id', 't0', 't1', 'duraton']].rename(columns={'vsetv_id': 'channel_id'})


def load_schedule(part):
    s = pd.read_csv(f'{D}/export_arh_{part}-final.csv', dtype={'year_of_production': str}, low_memory=False)
    s['t0'] = pd.to_datetime(s.start_time, format='%d.%m.%Y %H:%M:%S').astype('datetime64[s]').astype('int64')
    s['t1'] = s.t0 + s.duration
    s['part'] = part
    return s


def load_all_schedules():
    s = pd.concat([load_schedule(p) for p in SCHED], ignore_index=True)
    # the three files overlap at their boundaries -> drop duplicated airings
    s = s.drop_duplicates(['channel_id', 't0', 'tv_show_id']).sort_values(['channel_id', 't0']).reset_index(drop=True)
    return s


def join(views, sched, shift=0):
    """Return one row per (view, airing) overlap with overlap seconds.
    All airings (incl. tv_show_id=0) are joined; caller filters."""
    out = []
    for ch, sv in views.groupby('channel_id'):
        ss = sched[sched.channel_id == ch]
        if ss.empty:
            continue
        st = ss.t0.values + shift; en = ss.t1.values + shift
        # schedule ends are not strictly monotone (rare overlaps); use running max for the lower bound
        enmax = np.maximum.accumulate(en)
        lo = np.searchsorted(enmax, sv.t0.values, side='right')
        hi = np.searchsorted(st, sv.t1.values, side='left')
        n = np.maximum(hi - lo, 0)
        if n.sum() == 0:
            continue
        vi = np.repeat(np.arange(len(sv)), n)
        si = np.concatenate([np.arange(a, b) for a, b in zip(lo[n > 0], hi[n > 0])])
        ov = np.minimum(sv.t1.values[vi], en[si]) - np.maximum(sv.t0.values[vi], st[si])
        keep = ov > 0
        out.append(pd.DataFrame({'vidx': sv.index.values[vi][keep], 'sidx': ss.index.values[si][keep], 'overlap': ov[keep]}))
    return pd.concat(out, ignore_index=True)
