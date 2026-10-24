# sweet.tv TV-programme recommender — LLM4Rec final exam (AVITO)

Kaggle: [sweettv-tv-program-recommender](https://www.kaggle.com/competitions/sweettv-tv-program-recommender), metric MAP@5.

**Final submission file:** [`submissions/final_hybrid_lambdarank.csv`](submissions/final_hybrid_lambdarank.csv)
Other files in `submissions/`: `popularity_baseline.csv` (required baseline, same pipeline) and `hybrid_gbm.csv` (an early v1 built for a wrong, looser target definition before the official rule was found; kept as a record, not the final).

(final = hybrid: candidates from popularity / history / CBF / association rules / ALS / personalised PageRank, re-ranked by LightGBM LambdaRank).

## Validation results (official target, hold-out: history weeks 11-24 -> target weeks 25-30, 1,260 test users)

Target = for each user, the 5 programmes with the most airings watched with screen time >= 0.8
(rule published on the competition's Data page; re-implemented in `src/core.py::official_truth`,
checked against a brute-force recomputation in `analysis/check_truth_bruteforce.txt`).

| Method | MAP@5 |
| --- | ---: |
| Popularity baseline (distinct viewers with screen time >= 0.8, last 4 history weeks) | 0.2162 |
| Personal history (decayed screen-time sum, helper) | 0.3106 |
| Content-based filtering (TF-IDF over category/genres/director/actors/decade/channel) | 0.1914 |
| Association rules (user-week baskets, confidence, lift > 1) | 0.2005 |
| Matrix factorisation (implicit ALS, 8 factors) | 0.1695 |
| Personalised PageRank (item co-view graph, restart 0.3) | 0.3142 |
| **Hybrid: candidates + LightGBM LambdaRank re-ranker** | **0.3249** |

Raw outputs: `analysis/audit.txt` (table above, bootstrap CIs, label-shuffle control, cold-start and hand AP checks),
`analysis/segments.txt`, `analysis/select_labels.txt`, `analysis/weights_check.txt`, `analysis/als_grid.txt`,
`analysis/examples.txt`, `analysis/check_truth_bruteforce.txt`, `analysis/hybrid_run.txt`.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# put the 5 competition CSVs into data/ (kaggle competitions download -c sweettv-tv-program-recommender)
cd src
../.venv/bin/python build_interactions.py   # view log x schedule join -> data/user_airing.parquet (~20 s)
../.venv/bin/python evaluate.py val         # each method individually on the hold-out
../.venv/bin/python hybrid.py               # hybrid val score + submissions/final_hybrid_lambdarank.csv (~140 s)
../.venv/bin/python audit.py                # leakage / bootstrap / label-shuffle / cold-start / hand AP checks
../.venv/bin/python check_truth_bruteforce.py   # target rule recomputed from raw CSVs for 5 users
../.venv/bin/python weights_check.py        # interaction-weight choice
../.venv/bin/python select_labels.py        # re-ranker label / training-fold choice
../.venv/bin/python segments.py             # MAP by activity tercile, familiarity of lists
../.venv/bin/python examples.py             # two-user Top-5 tables
cd ../report && ../.venv/bin/python build_report.py   # report.html from analysis/ outputs
```

Environment: macOS 26.6.2, Apple M2 (8 cores, 8 GB), Python 3.12.13, versions in `requirements.txt`, `OPENBLAS_NUM_THREADS=1`. Seeds: 42 (ALS, LightGBM), 0 (bootstrap).

## Protocol summary

* **Join:** same channel (`vsetv_id = channel_id`) and view interval overlapping the airing interval `[start_time, start_time + duration)`; fragments of the same user-airing are summed. Timezone offsets of -3..+3 h were tested and 0 h fits best (`src/check_shift.py`).
* **Screen time** = watched seconds / airing duration (organisers' definition). Target: airings with screen time >= 0.8, top-5 programmes by count. History interactions: overlap >= min(300 s, 50 % of airing), weighted by screen time with an 8-week half-life.
* **`tv_show_id = 0`:** dropped as items (43.6 % of overlapped seconds); never recommended.
* **Candidate pool:** labelled programmes airing in the target window (schedule only); re-recommending watched programmes is allowed, identically for all methods.
* **Cold start:** users without any watched labelled airing in history get the popularity Top-5 in every method; ties/short lists are filled in popularity order.
* **No future views:** the log ends in week 30; the test-week schedule is used only for "what airs where and when".

src/ files `explore.py` and `check_shift.py` are the exploratory scripts used before the join.
