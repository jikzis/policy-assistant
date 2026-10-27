"""Run the evaluation questions through all three approaches and save the results for the website.

For every question and approach it records the answer, the policies used, response time and tokens, and:
- correct_policy: the answer relies on an expected policy (or, for out-of-scope questions, declines);
- unsupported:    the answer states something the policy database does not support, or answers a
                  question the database does not cover. Out-of-scope answers are flagged automatically;
                  LLM answers to covered questions are checked by a separate Claude "judge" call that sees
                  the whole policy database. Rules-based answers quote a policy verbatim, so they can be
                  wrong but cannot add facts.

usage: python evaluate.py            (needs ANTHROPIC_API_KEY in the environment or in .env)
"""
import json
import statistics
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


def score(item: dict, r: core.Result) -> dict:
    row = r.to_dict()
    if item["type"] == "out_of_scope":
        row["correct_policy"] = not r.found
        row["unsupported"] = r.found  # answering a question the database does not cover
        row["judge_note"] = "Answered an out-of-scope question" if r.found else ""
    else:
        row["correct_policy"] = any(p in r.policies for p in item["policies"])
        row["unsupported"], row["judge_note"] = False, ""
        if r.approach != "rules" and r.found:
            supported, note, judge_tokens = judge(item["question"], r.answer)
            row["unsupported"], row["judge_note"], row["judge_tokens"] = not supported, note, judge_tokens
    return row


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


def main():
    items = json.loads(QUESTIONS.read_text(encoding="utf8"))
    core.warm_up()
    rows = []
    for i, item in enumerate(items, 1):
        for key, fn in core.ANSWER.items():
            r = fn(item["question"])
            row = score(item, r) | {"question": item["question"], "type": item["type"],
                                     "expected": item["policies"]}
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
