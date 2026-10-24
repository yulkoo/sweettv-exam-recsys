# sweet.tv TV-programme recommender — LLM4Rec final exam (AVITO)

Kaggle: [sweettv-tv-program-recommender](https://www.kaggle.com/competitions/sweettv-tv-program-recommender), metric MAP@5.

**Final submission file:** [`submissions/final_hybrid_lambdarank.csv`](submissions/final_hybrid_lambdarank.csv)
Other files in `submissions/`: `popularity_baseline.csv` (required baseline, same pipeline) and `hybrid_gbm.csv` (an earlier v1: pointwise HistGradientBoosting re-ranker fitted on fit+val folds, produced before the tie/cold-start fix; kept as a record, not the final).

(final = hybrid: candidates from popularity / history / CBF / association rules / ALS / personalised PageRank, re-ranked by LightGBM LambdaRank).

## Validation results (hold-out: history weeks 11-24 -> target weeks 25-30, 1,260 test users)

| Method | MAP@5 |
| --- | ---: |
| Popularity baseline (distinct viewers, last 4 history weeks) | 0.6544 |
| Personal history (decayed watch hours) | 0.6055 |
| Content-based filtering (TF-IDF over category/genres/director/actors/decade/channel) | 0.3671 |
| Association rules (user-week baskets, confidence, lift > 1) | 0.6514 |
| Matrix factorisation (implicit ALS, 64 factors) | 0.4701 |
| Personalised PageRank (item co-view graph, restart 0.3) | 0.6558 |
| **Hybrid: candidates + LambdaRank re-ranker** | **0.7410** |

Raw outputs: `analysis/methods_val.txt`, `analysis/hybrid_run.txt`, `analysis/audit.txt`,
`analysis/robustness.txt`, `analysis/select_hybrid.txt`, `analysis/examples.txt`.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# put the 5 competition CSVs into data/ (kaggle competitions download -c sweettv-tv-program-recommender)
cd src
../.venv/bin/python build_interactions.py   # view log x schedule join -> data/user_airing.parquet (~20 s)
../.venv/bin/python evaluate.py val         # each method individually on the hold-out
../.venv/bin/python hybrid.py               # hybrid val score + submissions/final_hybrid_lambdarank.csv (~80 s)
../.venv/bin/python audit.py                # leakage / bootstrap / label-shuffle / cold-start / hand AP checks
../.venv/bin/python robustness.py           # sensitivity to the 'watched' threshold
../.venv/bin/python select_hybrid.py        # re-ranker variants compared on val
../.venv/bin/python examples.py             # two-user Top-5 tables
```

Environment: macOS 26.6.2, Apple M2 (8 cores, 8 GB), Python 3.12.13, versions in `requirements.txt`, `OPENBLAS_NUM_THREADS=1`. Seeds: 42 (ALS, LightGBM), 0 (bootstrap).

## Protocol summary

* **Join:** same channel (`vsetv_id = channel_id`) and view interval overlapping the airing interval `[start_time, start_time + duration)`; fragments of the same user-airing are summed. Timezone offsets of -3..+3 h were tested and 0 h fits best (`src/check_shift.py`).
* **Watched:** overlap >= min(300 s, 50 % of airing length).
* **`tv_show_id = 0`:** dropped as items (43.6 % of overlapped seconds); never recommended.
* **Candidate pool:** labelled programmes airing in the target window (schedule only); re-recommending watched programmes is allowed, identically for all methods.
* **Cold start:** users without any watched labelled airing in history get the popularity Top-5 in every method; ties/short lists are filled in popularity order.
* **No future views:** the log ends in week 30; the test-week schedule is used only for "what airs where and when".

src/ files `explore.py` and `check_shift.py` are the exploratory scripts used before the join.
