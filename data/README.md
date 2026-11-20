# The filings

Three Form 10-K annual reports from SEC EDGAR, each printed to PDF. They are public filings but are not stored in
this repo: put them in `data/pdfs/` with these names.

| File | Company | Fiscal year ended | Pages | sha256 of the copy used in `results/` |
| --- | --- | --- | --- | --- |
| `NVIDIA_FY2026_10-K.pdf` | NVIDIA | Jan 25, 2026 | 113 | `2e09e063d34ee38556f0bf5ad9cdf8ed6972871b82647e1974b3193d01566d57` |
| `AMD_FY2025_10-K.pdf` | AMD | Dec 27, 2025 | 148 | `e65b4dbb96d53daf02242fda935567dd7b864cdd2aa056fd90251c62ba0724e7` |
| `Intel_FY2025_10-K.pdf` | Intel | Dec 27, 2025 | 194 | `aa5f48f52ba6d06c8ab6dc30fd4bf82712151f415da227d2d2b6093c8b033bc8` |

To make your own copies from EDGAR:

```bash
pip install playwright && python3 -m playwright install chromium
SEC_USER_AGENT="Your Name you@example.com" python3 scripts/fetch_10k.py
```

SEC asks every client to identify itself with a name and email in the User-Agent
([fair access](https://www.sec.gov/os/accessing-edgar-data)).

Your copies may break pages in different places. That does not break the metrics: each question's evidence is a
text anchor (the figure itself, like `215,938`), and the pages that contain it are found when the benchmark starts.
Page numbers in `results/` refer to the copies above.

## The questions

`questions.json` holds 20 questions written for this benchmark: 8 easy (one figure, one filing), 5 medium (a change,
a driver or a share), 5 hard (the same figure across two or three filings) and 2 that the filings cannot answer.
Each answerable question has a reference answer and one evidence entry per fact it needs.
