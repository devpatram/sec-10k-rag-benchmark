"""Runs the benchmark: retrieve, answer, judge, summarize.

  python -m ragbench.run --retrieval-only     the three retrievers and the retrieval metrics (CPU is fine)
  python -m ragbench.run                      everything, including the generator and the judge (use a GPU)

Writes to results/ (or --out): answers.jsonl, summary.csv, by_difficulty.csv, summary.md, comparison.png.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import prompts
from .corpus import load_pages, evidence_pages
from .metrics import recall, reciprocal_rank
from .retrieve import (BM25Retriever, DecompositionRetriever, DenseRetriever, HybridRetriever,
                       parse_subquestions)

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "embedder": "BAAI/bge-small-en-v1.5",
    "generator": "Qwen/Qwen2.5-1.5B-Instruct",
    "judge": "HuggingFaceTB/SmolLM2-1.7B-Instruct",  # a different family from the generator
}


def load_questions(path: Path, pages) -> list[dict]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    for q in questions:
        q["groups"] = []
        for e in q["evidence"]:
            found = {(e["document"], p) for p in evidence_pages(pages, e["document"], e["anchor"])}
            if not found:
                raise ValueError(f"Question {q['id']}: anchor {e['anchor']!r} is on no page of {e['document']}.")
            q["groups"].append(found)
    return questions


def context(pages, ids, max_chars: int) -> str:
    return "\n\n".join(f"[Source: {pages[i].label()}]\n{pages[i].text[:max_chars]}" for i in ids)


def score(text: str) -> float:
    m = re.search(r"[1-5]", text)
    return float(m.group()) if m else np.nan


def one_line(text: str, limit: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def device() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except ImportError:
        return "CPU"


def summarize(df: pd.DataFrame, k: int) -> pd.DataFrame:
    answerable = df[df["difficulty"] != "unanswerable"]
    unanswerable = df[df["difficulty"] == "unanswerable"]
    table = answerable.groupby("configuration", sort=False).agg(
        MRR=("rr", "mean"), **{f"Recall@{k}": ("recall", "mean")})
    if "correctness" in df:
        table["Correctness (1-5)"] = answerable.groupby("configuration", sort=False)["correctness"].mean()
        table["Faithfulness (1-5)"] = df.groupby("configuration", sort=False)["faithfulness"].mean()
        table["Refused answerable"] = answerable.groupby("configuration", sort=False)["abstained"].sum().astype(int)
        table["Abstained on unanswerable"] = (unanswerable.groupby("configuration", sort=False)["abstained"].sum()
                                              .astype(int).astype(str) + f" of {len(unanswerable) // df['configuration'].nunique()}")
    return table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf-dir", default=ROOT / "data" / "pdfs", type=Path)
    ap.add_argument("--questions", default=ROOT / "data" / "questions.json", type=Path)
    ap.add_argument("--out", default=ROOT / "results", type=Path)
    ap.add_argument("--k", type=int, default=5, help="pages given to the generator and scored by Recall@k")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--gen-chars", type=int, default=2500, help="characters kept per page for the generator")
    ap.add_argument("--judge-chars", type=int, default=1200, help="characters kept per page for the judge")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--generator", default=MODELS["generator"], help="Hugging Face model ID")
    ap.add_argument("--judge", default=MODELS["judge"], help="Hugging Face model ID, ideally another family")
    ap.add_argument("--limit", type=int, help="only the first N questions plus the unanswerable ones (smoke test)")
    args = ap.parse_args(argv)
    MODELS.update(generator=args.generator, judge=args.judge)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    pages = load_pages(args.pdf_dir)
    questions = load_questions(args.questions, pages)
    if args.limit:
        questions = [q for q in questions if q["id"] <= args.limit or q["difficulty"] == "unanswerable"]
    print(f"{len(pages)} pages, {len(questions)} questions")

    dense = DenseRetriever(pages, MODELS["embedder"])
    bm25 = BM25Retriever(pages)
    retrievers = [dense, bm25, HybridRetriever(dense, bm25)]

    generator = judge = None
    if not args.retrieval_only:
        from .llm import ChatModel

        generator, judge = ChatModel(MODELS["generator"]), ChatModel(MODELS["judge"])
        print(f"Generator {generator.name} ({generator.parameters / 1e9:.2f}B), "
              f"judge {judge.name} ({judge.parameters / 1e9:.2f}B)")

    answers_cache: dict[tuple, str] = {}

    def run(retriever) -> list[dict]:
        rows = []
        for q in questions:
            ids = retriever.rank(q["question"], n=50)[: args.k]
            got = [(pages[i].document, pages[i].page) for i in ids]
            row = {"configuration": retriever.name, "id": q["id"], "difficulty": q["difficulty"],
                   "question": q["question"], "reference": q["answer"], "retrieved": [list(g) for g in got],
                   "rr": reciprocal_rank(got, q["groups"]) if q["groups"] else np.nan,
                   "recall": recall(got, q["groups"]) if q["groups"] else np.nan}
            if isinstance(retriever, DecompositionRetriever):
                row["subquestions"] = retriever.last_subquestions
            if generator is not None:
                key = (q["question"], tuple(ids))  # dense and hybrid often return the same pages
                if key not in answers_cache:
                    answers_cache[key] = generator(prompts.ANSWER.format(
                        context=context(pages, ids, args.gen_chars), question=q["question"]), args.max_new_tokens)
                answer = answers_cache[key]
                abstained = answer.strip().upper().startswith(prompts.ABSTAIN)
                row.update(answer=answer, abstained=abstained)
                # Abstention is scored in code: a small judge is unreliable at grading a refusal.
                if abstained:
                    row["correctness"] = 5.0 if not q["groups"] else 1.0
                    row["faithfulness"] = np.nan  # no claims to check
                else:
                    if q["groups"]:
                        row["correctness_raw"] = judge(prompts.CORRECTNESS.format(
                            question=q["question"], reference=q["answer"], answer=answer), 4)
                        row["correctness"] = score(row["correctness_raw"])
                    else:
                        row["correctness"] = 1.0  # answered a question the filings cannot answer
                    row["faithfulness_raw"] = judge(prompts.FAITHFULNESS.format(
                        context=context(pages, ids, args.judge_chars), answer=answer), 4)
                    row["faithfulness"] = score(row["faithfulness_raw"])
            rows.append(row)
        print(f"  {retriever.name}: done ({time.time() - started:.0f}s)")
        return rows

    rows = [r for retriever in retrievers for r in run(retriever)]
    df = pd.DataFrame(rows)
    base = summarize(df, args.k)

    if generator is not None:
        # Decomposition goes on top of the best base retriever: highest correctness, then Recall@k.
        best_name = base.sort_values(["Correctness (1-5)", f"Recall@{args.k}"], ascending=False).index[0]
        best = next(r for r in retrievers if r.name == best_name)
        split = lambda question: parse_subquestions(generator(prompts.SPLIT.format(question=question), 100))
        df = pd.concat([df, pd.DataFrame(run(DecompositionRetriever(best, split)))], ignore_index=True)

    summary = summarize(df, args.k)
    write_outputs(df, summary, args, pages, started)
    print(summary.round(3).to_string())
    return 0


def write_outputs(df: pd.DataFrame, summary: pd.DataFrame, args, pages, started: float) -> None:
    out, k = args.out, args.k
    with open(out / "answers.jsonl", "w", encoding="utf-8") as f:
        for row in df.to_dict("records"):
            f.write(json.dumps({key: (None if isinstance(v, float) and np.isnan(v) else v)
                                for key, v in row.items()}, ensure_ascii=False) + "\n")
    summary.round(3).to_csv(out / "summary.csv")
    lines = [f"# Results ({'retrieval only' if args.retrieval_only else 'full run'})", "",
             f"{len(pages)} pages, {df['id'].nunique()} questions, k = {k}. "
             f"Models: {', '.join(f'{r} {m}' for r, m in MODELS.items() if r == 'embedder' or not args.retrieval_only)}.",
             f"Run on {time.strftime('%Y-%m-%d')}, Python {platform.python_version()}, {device()}, "
             f"{time.time() - started:.0f} s.", "",
             "## Retrieval (answerable questions)" if args.retrieval_only else
             f"## Comparison (retrieval and correctness over the {df[df['difficulty'] != 'unanswerable']['id'].nunique()} "
             "answerable questions; faithfulness over answers that were not refusals)", "",
             summary.round(3).to_markdown(), ""]
    if "correctness" in df:
        by = (df.groupby(["configuration", "difficulty"], sort=False)[["correctness", "faithfulness", "recall"]]
              .mean().round(2).reset_index())
        by.to_csv(out / "by_difficulty.csv", index=False)
        lines += ["## By difficulty", "", by.to_markdown(index=False), ""]
        un = df[df["difficulty"] == "unanswerable"][["configuration", "id", "abstained", "answer"]].copy()
        un["answer"] = un["answer"].map(one_line)
        lines += ["## Unanswerable questions", "", un.to_markdown(index=False), ""]
        unparsed = sum(df[c].isna().sum() for c in ("correctness", "faithfulness")) - df["abstained"].sum()
        lines += [f"Judge replies without a score: {int(unparsed)} (left out of the means; raw replies are in "
                  "answers.jsonl).", ""]
        final_best = summary.sort_values(["Correctness (1-5)", f"Recall@{k}"], ascending=False).index[0]
        rows = df[(df["configuration"] == final_best) & (df["difficulty"] != "unanswerable")].copy()
        rows["stage"] = np.where(rows["recall"] < 1.0, "retrieval", "generation")
        worst = rows.sort_values(["correctness", "recall", "id"]).head(2).copy()
        worst["answer"] = worst["answer"].map(one_line)
        lines += [f"## Two weakest answers from the best configuration ({final_best})", "",
                  worst[["id", "question", "answer", "correctness", "recall", "stage"]].to_markdown(index=False), ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    from .chart import comparison_chart

    comparison_chart(summary, k, out / "comparison.png")
    print(f"Wrote {out}/summary.md and comparison.png")


if __name__ == "__main__":
    raise SystemExit(main())
