"""The prompts. Whether an answer abstained is scored in code, not by the judge (see run.py)."""

ABSTAIN = "NOT IN CONTEXT"

ANSWER = """You answer questions about company annual reports (Form 10-K) using only the excerpts below.
Each excerpt starts with its source document and page.

Rules:
- Use only figures that appear in the excerpts. Give each figure with its unit and the period it covers.
- When the question compares companies, give every company's figure, then the comparison.
- If the excerpts do not contain the answer, reply with exactly: NOT IN CONTEXT

Excerpts:
{context}

Question: {question}
Answer:"""

SPLIT = """Split the question below into 2 to 4 short search queries for a set of 10-K filings.
Each query should name one company and one figure or topic. Write one query per line and do not answer.

Question: {question}"""

CORRECTNESS = """You are grading an answer about company annual reports against a reference answer.

Question: {question}
Reference answer: {reference}
Answer to grade: {answer}

Score from 1 to 5:
5 = every figure and fact in the reference is present and right (rounding is fine)
4 = right, with one minor detail missing or imprecise
3 = partly right: at least one material figure or fact is wrong or missing
2 = mostly wrong, with little overlap with the reference
1 = wrong, or contradicts the reference

Reply with one digit only."""

FAITHFULNESS = """Check whether an answer is supported by the excerpts it was given. Do not judge whether it is
correct, only whether each of its claims can be found in the excerpts.

Excerpts:
{context}

Answer: {answer}

Score from 1 to 5:
5 = every claim is stated in the excerpts
4 = the main claims are stated, one minor detail is not
3 = some claims are stated and some are not
2 = most claims are not in the excerpts
1 = the answer is not supported by the excerpts at all

Reply with one digit only."""
