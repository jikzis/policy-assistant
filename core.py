"""The three ways of answering a policy question that the website compares and the Slack bot uses.

1. rules  - keyword search with hand-written rules (stop words, light stemming, synonyms); no LLM.
2. llm    - Claude with the whole policy database in the prompt; no vector index.
3. rag    - a vector index (bge-small embeddings) retrieves the top policies; only those go to Claude.
"""
import csv
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import anthropic
import numpy as np

DATA = Path(__file__).parent / "data" / "company_policies.csv"
MODEL = "claude-opus-5-5"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 3
NOT_FOUND = "No policy in the database covers this question."

APPROACHES = {
    "rules": "Rules-based search",
    "llm": "LLM without vector index",
    "rag": "LLM with vector index",
}


@dataclass
class Result:
    approach: str
    answer: str
    policies: list[str]           # titles of the policies the answer relies on
    found: bool                   # False when the approach says no policy covers the question
    seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    retrieved: list[tuple[str, float]] = field(default_factory=list)  # (title, score) shown for transparency

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        return asdict(self) | {"tokens": self.tokens}


@lru_cache
def policies() -> list[dict]:
    with open(DATA, encoding="utf8") as f:
        return list(csv.DictReader(f))


def by_title(title: str) -> dict | None:
    return next((p for p in policies() if p["title"].lower() == title.strip().lower()), None)


def policy_line(p: dict) -> str:
    return f"[{p['title']}] (Category: {p['category']}; Department: {p['department']}) {p['policy_text']}"


# ---------------------------------------------------------------- 1. rules-based search

STOP_WORDS = set("""
a an and are as at be been by can could do does did for from get got have has how i if in into is it its
may me must my of on or our should so than that the their them then there these they this to up us was we
what when where which who whom why will with would you your about any all am also policy policies company
employee employees tell know need want please there's what's whats allowed rule rules many much long often
work day days time staff people
""".split())

# a few hand-written synonym rules: word in the question -> words used in the policies
SYNONYMS = {
    "home": ["remote"], "wfh": ["remote"], "telework": ["remote"], "remotely": ["remote"],
    "die": ["bereavement"], "dies": ["bereavement"], "died": ["bereavement"], "death": ["bereavement"],
    "funeral": ["bereavement"], "passed": ["bereavement"],
    "holiday": ["vacation", "holiday"], "pto": ["vacation"], "leave": ["leave", "vacation"],
    "sick": ["sick"], "ill": ["sick"], "illness": ["sick"],
    "baby": ["parental"], "maternity": ["parental"], "paternity": ["parental"],
    "buy": ["purchase", "procurement", "expense"], "buying": ["purchase", "procurement", "expense"],
    "spend": ["expense"], "cost": ["expense"], "money": ["expense"], "dollars": ["expense"],
    "reimburse": ["reimbursement", "reimbursed"], "refund": ["reimbursement", "return"],
    "clothes": ["dress", "attire"], "wear": ["dress", "attire", "worn"],
    "laptop": ["laptop"], "computer": ["laptop", "hardware"], "phone": ["mobile", "device"],
    "hack": ["security", "incident"], "breach": ["security", "incident"], "phishing": ["security", "cybersecurity"],
    "car": ["vehicle"], "drive": ["vehicle"], "fired": ["exit"], "quit": ["exit", "departing"],
    "leaving": ["exit", "departing"], "resign": ["exit", "departing"], "hire": ["hires", "onboarding"],
    "abroad": ["abroad"], "overseas": ["abroad"], "travel": ["travel"], "trip": ["travel"],
    "press": ["media", "press"], "journalist": ["media"], "reporter": ["media"],
    "school": ["tuition"], "degree": ["tuition"], "course": ["training", "tuition"],
}


def stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def terms(text: str, expand: bool = False) -> set[str]:
    """Content words of text, stemmed; expand=True also adds the synonym rules (used for questions only)."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = set()
    for w in words:
        if w in STOP_WORDS:
            continue
        out.add(stem(w))
        if expand:
            out.update(stem(s) for s in SYNONYMS.get(w, SYNONYMS.get(stem(w), [])))
    return out


@lru_cache
def rule_index() -> list[tuple[dict, set[str], set[str]]]:
    return [(p, terms(p["title"]), terms(p["policy_text"] + " " + p["category"])) for p in policies()]


def answer_rules(question: str) -> Result:
    start = time.perf_counter()
    q = terms(question, expand=True)
    scored = []
    for p, title_terms, body_terms in rule_index():
        # rule: a match in the title counts three times as much as a match in the text
        title_hits = len(q & title_terms)
        scored.append((3 * title_hits + len(q & body_terms), title_hits, p))
    scored.sort(key=lambda s: (-s[0], -s[1]))  # ties go to the policy with more title matches
    best_score, _, best = scored[0]
    retrieved = [(p["title"], float(s)) for s, _, p in scored[:TOP_K] if s > 0]
    seconds = time.perf_counter() - start
    # rule: needs at least one title match (score >= 3), otherwise the question is treated as not covered
    if best_score < 3:
        return Result("rules", NOT_FOUND, [], False, seconds, retrieved=retrieved)
    return Result("rules", f"{best['title']}: {best['policy_text']}", [best["title"]], True, seconds,
                  retrieved=retrieved)


# ---------------------------------------------------------------- LLM shared

SYSTEM = """You answer employees' questions about company policy using ONLY the policies provided below.

Rules:
- Base the answer strictly on the policy text. Do not add numbers, durations, conditions or procedures
  that the policy does not state, and do not use general knowledge about how companies usually work.
- If a relevant policy exists but does not contain the detail asked for, say what the policy does say
  and that it does not specify the detail.
- If no provided policy covers the question, set found to false, leave policies empty and answer exactly:
  "No policy in the database covers this question."
- policies must list the exact titles (as written in square brackets) of the policies the answer relies on.
- Keep the answer to one to three sentences."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "policies": {"type": "array", "items": {"type": "string"}},
        "found": {"type": "boolean"},
    },
    "required": ["answer", "policies", "found"],
    "additionalProperties": False,
}


def api_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY")


@lru_cache
def client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key())


def call_claude(system: str, question: str, schema: dict = ANSWER_SCHEMA, effort: str = "low"):
    response = client().beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=system,
        messages=[{"role": "user", "content": question}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined to answer this question.")
    text = next((b.text for b in response.content if b.type == "text"), "{}")
    return json.loads(text), response.usage


def llm_result(approach: str, data: dict, usage, start: float, retrieved=()) -> Result:
    titles = [t for t in data.get("policies", []) if by_title(t)]  # drop titles that are not in the database
    found = bool(data.get("found")) and bool(titles)
    return Result(approach, data.get("answer", "").strip() or NOT_FOUND, titles, found,
                  time.perf_counter() - start, usage.input_tokens, usage.output_tokens, list(retrieved))


# ---------------------------------------------------------------- 2. LLM without vector index

def answer_llm(question: str) -> Result:
    start = time.perf_counter()
    context = "\n".join(policy_line(p) for p in policies())
    data, usage = call_claude(f"{SYSTEM}\n\nPolicies ({len(policies())}):\n{context}", question)
    return llm_result("llm", data, usage, start)


# ---------------------------------------------------------------- 3. LLM with vector index

@lru_cache
def embedder():
    from fastembed import TextEmbedding  # imported lazily: only needed for this approach
    return TextEmbedding(EMBED_MODEL)


@lru_cache
def vector_index() -> np.ndarray:
    docs = [f"{p['title']}. {p['policy_text']} (Category: {p['category']})" for p in policies()]
    return np.array(list(embedder().embed(docs)))  # bge embeddings are normalised: dot product = cosine


def retrieve(question: str, k: int = TOP_K) -> list[tuple[dict, float]]:
    q = np.array(list(embedder().query_embed([question])))[0]
    scores = vector_index() @ q
    top = np.argsort(-scores)[:k]
    return [(policies()[i], float(scores[i])) for i in top]


def answer_rag(question: str) -> Result:
    start = time.perf_counter()
    hits = retrieve(question)
    context = "\n".join(policy_line(p) for p, _ in hits)
    data, usage = call_claude(f"{SYSTEM}\n\nMost relevant policies (top {TOP_K} by vector search):\n{context}",
                              question)
    return llm_result("rag", data, usage, start, [(p["title"], round(s, 3)) for p, s in hits])


ANSWER = {"rules": answer_rules, "llm": answer_llm, "rag": answer_rag}


def warm_up():
    """Load the policy data and the vector index so the first question is not slowed down."""
    rule_index()
    vector_index()
