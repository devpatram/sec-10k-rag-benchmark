# Results (retrieval only)

455 pages, 20 questions, k = 5. Models: embedder BAAI/bge-small-en-v1.5.
Run on 2026-11-20, 3.13.17, 52 s.

## Comparison (answerable questions; faithfulness over answers that were not refusals)

| configuration   |   MRR |   Recall@5 |
|:----------------|------:|-----------:|
| Dense           | 0.417 |      0.389 |
| BM25            | 0.479 |      0.667 |
| Hybrid RRF      | 0.432 |      0.528 |
