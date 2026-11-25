"""Research tools: a searchable document corpus (BM25) and a company registry."""

from __future__ import annotations

import difflib
import json
import math
import re
from collections import Counter
from pathlib import Path

from .models import Evidence

STOP = frozenset(
    "a an the and or of to in on for with by at from as is are was were be been it its this that what which who "
    "how does do did has have had any there their our about into over not no".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")
_SUFFIX = re.compile(r"\b(inc|llc|ltd|co|corp|company)\b\.?", re.I)


def _stem(t: str) -> str:
    """Tiny plural stemmer: lawsuits -> lawsuit, reviews -> review, liens -> lien."""
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def words(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOP and len(t) > 1]


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", _SUFFIX.sub("", name.lower())).strip()


class Corpus:
    def __init__(self, root: str | Path):
        self.docs: list[Evidence] = []
        self.company_of: dict[str, str] = {}
        for p in sorted(Path(root).rglob("*.md")):
            raw = p.read_text(encoding="utf-8")
            title = raw.splitlines()[0].lstrip("# ").strip()
            body = "\n".join(line for line in raw.splitlines()[1:] if line.strip() and not line.startswith(">"))
            sid = f"{p.parent.name}/{p.stem}"
            self.docs.append(Evidence(source_id=sid, title=title, text=body.strip()))
            self.company_of[sid] = p.parent.name
        self._tf = [Counter(words(d.title + " " + d.text)) for d in self.docs]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / max(1, len(self._len))
        df = Counter(t for tf in self._tf for t in tf)
        n = len(self.docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def get(self, source_id: str) -> Evidence | None:
        return next((d for d in self.docs if d.source_id == source_id), None)

    def search(self, query: str, company_slug: str | None = None, k: int = 3) -> list[Evidence]:
        q = words(query)
        scored = []
        for i, (tf, dl) in enumerate(zip(self._tf, self._len, strict=True)):
            if company_slug and self.company_of[self.docs[i].source_id] != company_slug:
                continue
            s = sum(self._idf[t] * tf[t] * 2.5 / (tf[t] + 1.5 * (0.25 + 0.75 * dl / self._avg)) for t in q if t in tf)
            if s > 0:
                scored.append((s, i))
        return [self.docs[i] for _, i in sorted(scored, reverse=True)[:k]]


class Registry:
    def __init__(self, path: str | Path):
        self.records = json.loads(Path(path).read_text())

    def lookup(self, name: str) -> dict | None:
        target = normalize_name(name)
        best = max(self.records, key=lambda r: difflib.SequenceMatcher(None, target, normalize_name(r["name"])).ratio())
        return best if difflib.SequenceMatcher(None, target, normalize_name(best["name"])).ratio() >= 0.8 else None

    def as_evidence(self, record: dict) -> Evidence:
        text = (
            f"{record['name'].rstrip('.')} was incorporated on {record['incorporated']} in {record['state']}. "
            f"Registry status is {record['status']}. Sanctions list match: "
            f"{'yes' if record['sanctions_match'] else 'no'}. "
            f"Registered agent: {record['registered_agent'].rstrip('.')}."
        )
        return Evidence(
            source_id=f"registry/{record['slug']}", title=f"Company registry record: {record['name']}", text=text
        )


def support_score(claim: str, evidence: list[Evidence]) -> float:
    """Share of the claim's content words that appear in the cited evidence."""
    w = words(claim)
    if not w or not evidence:
        return 0.0
    vocab = set()
    for e in evidence:
        vocab.update(words(e.title + " " + e.text))
    return sum(x in vocab for x in w) / len(w)
