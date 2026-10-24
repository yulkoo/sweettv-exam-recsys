"""Independent re-computation of the official-style target for a few users straight from the raw CSVs
(plain loops, no shared join code), compared with core.official_truth."""
import csv, datetime as dt, random, collections
from core import *
FMT_V = '%Y-%m-%d %H:%M:%S'; FMT_S = '%d.%m.%Y %H:%M:%S'
ts = lambda s, f: int(dt.datetime.strptime(s, f).replace(tzinfo=dt.timezone.utc).timestamp())
lo, hi = week_start(25), week_start(31)
g, s, users = load()
truth = Fold('val', g, s, users).truth
random.seed(7); sample = set(random.sample(sorted(truth), 5))
views = collections.defaultdict(list)
for r in csv.DictReader(open(f'{D}/dataset11-30.csv')):
    u = int(r['user_id'])
    if u in sample:
        views[u].append((int(r['vsetv_id']), ts(r['start_time'], FMT_V), ts(r['stop_time'], FMT_V)))
air = collections.defaultdict(list)          # channel -> airings starting in wk 25-30
for p in ['21-30', '31-42']:
    for r in csv.DictReader(open(f'{D}/export_arh_{p}-final.csv')):
        if r['tv_show_id'] != '0':
            t0 = ts(r['start_time'], FMT_S)
            if lo <= t0 < hi:
                air[int(r['channel_id'])].append((t0, t0 + int(r['duration']), int(r['tv_show_id'])))
for u in sorted(sample):
    seen = collections.defaultdict(int)       # (channel, airing start, id) -> watched seconds
    for ch, a, b in views[u]:
        for t0, t1, i in air.get(ch, []):
            ov = min(b, t1) - max(a, t0)
            if ov > 0: seen[(ch, t0, t1, i)] += ov
    cnt = collections.Counter(); sec = collections.Counter()
    for (ch, t0, t1, i), v in seen.items():
        v = min(v, t1 - t0)
        if v / (t1 - t0) >= 0.8: cnt[i] += 1; sec[i] += v
    top = sorted(cnt, key=lambda i: (-cnt[i], -sec[i], i))[:5]
    print(u, 'brute-force', sorted(top), '| pipeline', sorted(truth[u]), '| equal:', set(top) == truth[u])
