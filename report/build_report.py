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

# Table II: interaction weight (half-life 8) + re-ranker label/folds
rows = []
for l in open(f'{A}/weights_check.txt'):
    p = [x.strip() for x in l.split('|')]
    if p[0] == 'half-life 8 wk':
        best = 'screen-time' in p[1]
        rows.append(f'<tr{" class=best" if best else ""}><td>History weight: {html.escape(p[1])}{" (used)" if best else ""}</td><td class="n">{p[2].split("= ")[1]}</td><td class="n">—</td></tr>')
sf = open(f'{A}/select_final.txt').read()
def sfv(label):
    m = re.search(re.escape(label) + r'\s+val ([\d.]+)(?:\s+fit ([\d.]+))?', sf)
    return m.group(1), m.group(2) or '—'
SEL = [('Re-ranker: first setting (63 leaves), trained on 2 folds', 'old setting, fit+fit2'),
       ('… same, trained on 5 folds', 'old setting, 5 folds'),
       ('… + personal (channel, hour) habit features per programme', 'old setting + own-slot features'),
       ("… 31 leaves, &ge; 300 rows/leaf", "{'num_leaves': 31, 'min_child_samples': 300}"),
       ("… rank_xendcg objective", "{'objective': 'rank_xendcg'}"),
       ('… 31 leaves, &ge; 300 rows/leaf, 5-seed bag (final)', 'num_leaves=31, min_child_samples=300, 5 seeds  (FINAL)'),
       ('Labels from 3-week windows (val window is 6 weeks)', '3-week labels')]
for name, key in SEL:
    v, f2 = sfv(key)
    rows.append(f'<tr{" class=best" if "final" in name else ""}><td>{name}</td><td class="n">{v}</td><td class="n">{f2}</td></tr>')
rp = re.findall(r"'rank\+poisson': ([\d.]+)", sf)          # [val, fit]
rows.append(f'<tr><td>Final + pointwise Poisson LightGBM (rank average)</td><td class="n">{rp[0]}</td><td class="n">{rp[1]}</td></tr>')
t = t.replace('{{TABLE2}}', '\n'.join(rows))
RECALL = re.search(r'candidate recall of val truth: ([\d.]+)', open(f'{A}/select_labels.txt').read()).group(1)   # same candidates
t = t.replace('{{RECALL}}', f'{float(RECALL):.1%}'.replace('%', '%'))
sub = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'submissions', 'final_hybrid_lambdarank.csv'))
t = t.replace('{{EX_ROW0}}', sub.tv_show_id[sub.user_id == 8377619604347126107].iloc[0])

# Examples
s = pd.read_parquet(f'{D}/schedule.parquet'); s = s[s.tv_show_id != 0]
title = s.groupby('tv_show_id').tv_show_title.agg(lambda x: x.value_counts().index[0])
cat = s.groupby('tv_show_id').tv_show_category.agg(lambda x: x.value_counts().index[0])
def cell(i, hit=False, seen=True):
    tt = title.get(i, '?').rstrip(' .')
    tt = html.escape(tt if len(tt) <= 32 else tt[:31].rstrip() + '…')
    return f'<td{" class=hit" if hit else ""}>{i}<br>{tt} ({cat.get(i, "?")}){"" if seen else " <i>new</i>"}</td>'
ex = json.load(open(f'{A}/examples.json'))
out = []
for lab, e in ex.items():
    out.append(f'<table class="ex"><tr><th style="width:13%">{lab.capitalize()} viewer <code>{e["user"]}</code><br>{e["hours"]:.1f} h, '
               f'{e["n_hist_items"]} progr. in wk 11–24; official top-{e["n_truth"]} in wk 25–30</th>'
               + ''.join(f'<th>#{k}</th>' for k in range(1, 6)) + '<th class="n">AP@5</th></tr>')
    out.append('<tr><td><i>History top-5 by hours</i></td>' + ''.join(cell(i) for i in e['history_top']) + '<td></td></tr>')
    for m, r in e['val'].items():
        out.append(f'<tr><td>{NAMES[m]}</td>' + ''.join(cell(i, h, sn) for i, h, sn in zip(r['ids'], r['hits'], r['seen']))
                   + f'<td class="n">{r["ap5"]:.2f}</td></tr>')
    out.append('<tr><td><i>Final hybrid, test wk 31–42</i></td>' + ''.join(cell(i) for i in e['test_hybrid']) + '<td>—</td></tr>')
    out.append('</table>')
out.append('<div class="cap">Table III. Each method\'s Top-5 for the two users on the hold-out (id, most frequent schedule title, cut to 32 characters with “…”, category); bold = watched in wk 25–30.</div>')
t = t.replace('{{EXAMPLES}}', '\n'.join(out))
assert '{{' not in t
open(os.path.join(os.path.dirname(__file__), 'report.html'), 'w').write(t)
print('report.html written')
