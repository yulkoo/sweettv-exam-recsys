"""Fill report_src.html tables from the saved outputs in analysis/ -> report.html (no hand-copied numbers)."""
import json, re, html, pandas as pd, os
A = os.path.join(os.path.dirname(__file__), '..', 'analysis')
D = os.path.join(os.path.dirname(__file__), '..', 'data')
t = open(os.path.join(os.path.dirname(__file__), 'report_src.html')).read()
NAMES = {'popularity': 'Popularity (baseline)', 'history': 'Personal history (helper)', 'cbf': 'Content-based filtering',
         'assoc_rules': 'Association rules', 'als': 'Matrix factorization (ALS)', 'ppr': 'Personalized PageRank',
         'hybrid': 'Hybrid: candidates + LambdaRank'}

# Table I: overall from audit.txt, terciles + seen share from segments.txt
overall = dict(re.findall(r'^(\w+)\s+MAP@5=([\d.]+)$', open(f'{A}/audit.txt').read(), re.M))
seg = {}
for line in open(f'{A}/segments.txt'):
    p = line.split()
    if p and p[0] in NAMES and len(p) >= 7 and p[1][0].isdigit():
        seg[p[0]] = p[1:5]
rows = []
for m in NAMES:
    cls = ' class="best"' if m == 'hybrid' else ''
    l, md, h, seen = seg[m]
    rows.append(f'<tr{cls}><td>{NAMES[m]}</td><td class="n">{float(overall[m]):.4f}</td><td class="n">{l}</td><td class="n">{md}</td>'
                f'<td class="n">{h}</td><td class="n">{float(seen):.3f}</td></tr>')
t = t.replace('{{TABLE1}}', '\n'.join(rows))

# Table II
rows = []
for k, v in re.findall(r'^(.+?)\s+val MAP@5 = ([\d.]+)$', open(f'{A}/select_hybrid.txt').read(), re.M):
    k = k.replace('HGB pointwise', 'HistGradientBoosting, pointwise').replace('LGBM lambdarank', 'LightGBM LambdaRank') \
         .replace(', fit+fit2', ', fitted on FOLDS2').replace(', fit', ', fitted on fit').replace('FOLDS2', 'fit + fit2')
    cls = ' class="best"' if k == 'LightGBM LambdaRank, fitted on fit' else ''
    rows.append(f'<tr{cls}><td>{k}{" (final)" if cls else ""}</td><td class="n">{v}</td></tr>')
t = t.replace('{{TABLE2}}', '\n'.join(rows))

# Table III
lines = [l for l in open(f'{A}/robustness.txt') if re.search(r'\d\.\d{4}', l)]
rows = []
for l in lines:
    nums = re.findall(r'\d\.\d{4}', l); name = l[:l.index(nums[0])].strip()
    hy, pop, hist, cbf, ar, als, ppr = nums
    rows.append(f'<tr><td>{html.escape(name)}</td>' + ''.join(f'<td class="n">{x}</td>' for x in [hy, pop, ar, als, ppr, cbf]) + '</tr>')
t = t.replace('{{TABLE3}}', '\n'.join(rows))

# Examples
s = pd.read_parquet(f'{D}/schedule.parquet'); s = s[s.tv_show_id != 0]
title = s.groupby('tv_show_id').tv_show_title.agg(lambda x: x.value_counts().index[0])
cat = s.groupby('tv_show_id').tv_show_category.agg(lambda x: x.value_counts().index[0])
def cell(i, hit=False, seen=True):
    tt = html.escape(title.get(i, '?').strip(" .'")[:30])
    return f'<td{" class=hit" if hit else ""}>{i}<br>{tt} ({cat.get(i, "?")}){"" if seen else " <i>new</i>"}</td>'
ex = json.load(open(f'{A}/examples.json'))
out = []
for lab, e in ex.items():
    out.append(f'<table class="ex"><tr><th style="width:13%">{lab.capitalize()} viewer <code>{e["user"]}</code><br>{e["hours"]:.1f} h, '
               f'{e["n_hist_items"]} progr. in wk 11–24; {e["n_truth"]} in wk 25–30</th>'
               + ''.join(f'<th>#{k}</th>' for k in range(1, 6)) + '<th class="n">AP@5</th></tr>')
    out.append('<tr><td><i>History top-5 by hours</i></td>' + ''.join(cell(i) for i in e['history_top']) + '<td></td></tr>')
    for m, r in e['val'].items():
        out.append(f'<tr><td>{NAMES[m]}</td>' + ''.join(cell(i, h, sn) for i, h, sn in zip(r['ids'], r['hits'], r['seen']))
                   + f'<td class="n">{r["ap5"]:.2f}</td></tr>')
    out.append('<tr><td><i>Final hybrid, test wk 31–42</i></td>' + ''.join(cell(i) for i in e['test_hybrid']) + '<td>—</td></tr>')
    out.append('</table>')
out.append('<div class="cap">Table IV. Each method\'s Top-5 for the two users on the hold-out (id, most frequent title, category); bold = watched in wk 25–30.</div>')
t = t.replace('{{EXAMPLES}}', '\n'.join(out))
assert '{{' not in t
open(os.path.join(os.path.dirname(__file__), 'report.html'), 'w').write(t)
print('report.html written')
