"""Run the evaluation questions through all three approaches and save the results for the website.

For every question and approach it records the answer, the policies used, response time and tokens, and:
- correct_policy: the answer relies on an expected policy (or, for out-of-scope questions, declines);
- judge_flag:     a separate Claude "judge" call, which sees the whole policy database, found claims in an
                  LLM answer that the database does not support (rules-based answers quote a policy
                  verbatim, so they cannot add claims and are not judged);
- unsupported:    the approach gave an answer (did not decline) that is not backed by the database: it
                  relies on a policy that does not cover the question, answers an out-of-scope question,
                  or was flagged by the judge. The same definition applies to all three approaches.

usage: python evaluate.py            (needs ANTHROPIC_API_KEY in the environment or in .env)
       python evaluate.py --quick    (5 questions of mixed types, saved separately, to check the setup)
       python evaluate.py --rescore  (recompute scores and summary of the saved results; no API calls)
"""
import json
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import core  # noqa: E402  (core reads the API key from the environment)

QUESTIONS = Path(__file__).parent / "data" / "eval_questions.json"
OUT = Path(__file__).parent / "data" / "eval_results.json"

JUDGE_SYSTEM = """You check whether an answer to an employee's question is fully supported by a company's
policy database. A claim is supported only if it is stated in, or follows directly from, the policy text.
Saying that a policy does not specify a detail, or that no policy covers the question, is supported.
Numbers, limits, tools, procedures or conditions that are not in the policy text are NOT supported.

Policy database:
{policies}"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"supported": {"type": "boolean"}, "unsupported_claims": {"type": "string"}},
    "required": ["supported", "unsupported_claims"],
    "additionalProperties": False,
}


def judge(question: str, answer: str) -> tuple[bool, str, int]:
    system = JUDGE_SYSTEM.format(policies="\n".join(core.policy_line(p) for p in core.policies()))
    data, usage = core.call_claude(system, f"Question: {question}\nAnswer: {answer}", JUDGE_SCHEMA, "medium")
    return data["supported"], data["unsupported_claims"], usage.input_tokens + usage.output_tokens


def apply_scores(row: dict) -> dict:
    """correct_policy and unsupported from the answer, the expected policies and the judge flag."""
    if row["type"] == "out_of_scope":
        row["correct_policy"] = not row["found"]
    else:
        row["correct_policy"] = any(p in row["policies"] for p in row["expected"])
    row["unsupported"] = row["found"] and (not row["correct_policy"] or row.get("judge_flag", False))
    if row["unsupported"] and not row.get("judge_note"):
        row["judge_note"] = ("Answered an out-of-scope question" if row["type"] == "out_of_scope"
                             else "Answered with a policy that does not cover the question")
    return row


def score(item: dict, r: core.Result) -> dict:
    row = r.to_dict() | {"question": item["question"], "type": item["type"], "expected": item["policies"],
                         "judge_flag": False, "judge_note": ""}
    if r.approach != "rules" and r.found and item["type"] != "out_of_scope":
        supported, note, judge_tokens = judge(item["question"], r.answer)
        row |= {"judge_flag": not supported, "judge_note": note if not supported else "",
                "judge_tokens": judge_tokens}
    return apply_scores(row)


def rescore():
    data = json.loads(OUT.read_text(encoding="utf8"))
    for row in data["rows"]:
        if "judge_flag" not in row:  # results saved before judge_flag existed
            row["judge_flag"] = row["approach"] != "rules" and row["type"] != "out_of_scope" and row["unsupported"]
        row["judge_note"] = row["judge_note"] if row["judge_flag"] else ""
        apply_scores(row)
    data["summary"] = summarise(data["rows"])
    OUT.write_text(json.dumps(data, indent=2), encoding="utf8")
    print(json.dumps(data["summary"], indent=2))


def summarise(rows: list[dict]) -> dict:
    out = {}
    for key in core.APPROACHES:
        mine = [r for r in rows if r["approach"] == key]
        out[key] = {
            "questions": len(mine),
            "correct_policy_pct": round(100 * sum(r["correct_policy"] for r in mine) / len(mine), 1),
            "unsupported_pct": round(100 * sum(r["unsupported"] for r in mine) / len(mine), 1),
            "unsupported_count": sum(r["unsupported"] for r in mine),
            "median_seconds": round(statistics.median(r["seconds"] for r in mine), 3),
            "mean_seconds": round(statistics.mean(r["seconds"] for r in mine), 3),
            "mean_tokens": round(statistics.mean(r["tokens"] for r in mine)),
            "total_tokens": sum(r["tokens"] for r in mine),
        }
        for t in ("answerable", "detail_missing", "out_of_scope"):
            sub = [r for r in mine if r["type"] == t]
            out[key][f"correct_{t}"] = f"{sum(r['correct_policy'] for r in sub)}/{len(sub)}"
            out[key][f"unsupported_{t}"] = f"{sum(r['unsupported'] for r in sub)}/{len(sub)}"
    return out


def quick_sample(items: list[dict]) -> list[dict]:
    """2 answerable, 2 detail-missing and 1 out-of-scope question."""
    by_type = {t: [i for i in items if i["type"] == t] for t in ("answerable", "detail_missing", "out_of_scope")}
    return by_type["answerable"][:2] + by_type["detail_missing"][:2] + by_type["out_of_scope"][:1]


def main():
    global OUT
    if "--rescore" in sys.argv:
        return rescore()
    items = json.loads(QUESTIONS.read_text(encoding="utf8"))
    if "--quick" in sys.argv:
        items, OUT = quick_sample(items), OUT.with_name("eval_results_quick.json")
    if not core.api_key():
        sys.exit("ANTHROPIC_API_KEY is not set (add it to .env).")
    core.warm_up()
    rows = []
    for i, item in enumerate(items, 1):
        for key, fn in core.ANSWER.items():
            r = fn(item["question"])
            row = score(item, r)
            rows.append(row)
            print(f"{i:2}/{len(items)} {key:5} {'OK ' if row['correct_policy'] else 'BAD'} "
                  f"{'UNSUPPORTED' if row['unsupported'] else '':11} {r.seconds:5.2f}s {r.tokens:5} tok  "
                  f"{item['question'][:50]}")
    result = {"model": core.MODEL, "embedding_model": core.EMBED_MODEL, "top_k": core.TOP_K,
              "run_at": time.strftime("%Y-%m-%d %H:%M"), "summary": summarise(rows), "rows": rows}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
