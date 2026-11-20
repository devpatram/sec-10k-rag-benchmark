"""Downloads the three 10-K filings from SEC EDGAR and prints each one to a PDF, one file per company.

The question set's page numbers refer to the PDFs in data/pdfs (sha256 in data/README.md). A different Chromium
version can move page breaks, so check the hashes before you compare your numbers with results/.

Usage:  SEC_USER_AGENT="Your Name you@example.com" python3 scripts/fetch_10k.py
SEC asks every client to identify itself: https://www.sec.gov/os/accessing-edgar-data
Needs:  pip install playwright && python3 -m playwright install chromium
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "pdfs"
# Company, SEC CIK, fiscal year-end of the filing, output name.
FILINGS = [
    ("NVIDIA", "0001045810", "2026-01-25", "NVIDIA_FY2026_10-K"),
    ("AMD", "0000002488", "2025-12-27", "AMD_FY2025_10-K"),
    ("Intel", "0000050863", "2025-12-27", "Intel_FY2025_10-K"),
]


def get(url: str, agent: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": agent})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def filing_url(cik: str, period: str, agent: str) -> str:
    """The primary document of the 10-K for that fiscal year-end, from EDGAR's submissions index."""
    recent = json.loads(get(f"https://data.sec.gov/submissions/CIK{cik}.json", agent))["filings"]["recent"]
    for form, report, acc, doc in zip(recent["form"], recent["reportDate"], recent["accessionNumber"],
                                      recent["primaryDocument"]):
        if form == "10-K" and report == period:
            return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}"
    raise SystemExit(f"No 10-K for period {period} in CIK {cik}.")


def main() -> int:
    agent = os.getenv("SEC_USER_AGENT")
    if not agent:
        print("Set SEC_USER_AGENT to your name and email first (SEC's fair-access rule).")
        return 1
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        for company, cik, period, name in FILINGS:
            url = filing_url(cik, period, agent)
            html = OUT / f"{name}.htm"
            html.write_bytes(get(url, agent))
            page.goto(html.resolve().as_uri(), wait_until="load")
            page.pdf(path=str(OUT / f"{name}.pdf"), format="Letter")
            html.unlink()
            print(f"{company}: {url} -> data/pdfs/{name}.pdf")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
