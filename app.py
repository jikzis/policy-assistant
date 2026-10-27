"""Company Policy Assistant: compare rules-based search, an LLM without a vector index and an LLM with one."""
import hmac
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="Policy Assistant", page_icon="📘", layout="wide")

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")  # local runs; on Streamlit Cloud the values come from Secrets
RESULTS = HERE / "data" / "eval_results.json"
WRITEUP = HERE / "WRITEUP.md"


# ---------------------------------------------------------------- secrets and access

def secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets file locally
        value = None
    return value or os.environ.get(name)


if secret("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = secret("ANTHROPIC_API_KEY")  # core reads the key from the environment

import core  # noqa: E402


def require_password():
    """Stop the script until the viewer has entered the app password (fails closed if none is set)."""
    if st.session_state.get("authenticated"):
        return
    expected = secret("APP_PASSWORD")
    st.title("📘 Company Policy Assistant")
    if not expected:
        st.error("This app is locked: no `APP_PASSWORD` is configured in the app's secrets.")
        st.stop()
    with st.form("login"):
        entered = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Open", type="primary")
    if submitted:
        if hmac.compare_digest(entered.encode(), expected.encode()):
            st.session_state["authenticated"] = True
            st.rerun()
        time.sleep(2)  # slow down guessing
        st.error("Wrong password.")
    st.stop()


require_password()


@st.cache_resource(show_spinner="Loading the policy database and vector index…")
def warm_up():
    core.warm_up()
    return True


warm_up()


# ---------------------------------------------------------------- live comparison

EXAMPLES = [
    "How many days a week can I work from home?",
    "My grandmother passed away. How much time off can I take?",
    "How many sick days do I earn each month?",
    "Can I bring my dog to the office?",
]

DESCRIPTIONS = {
    "rules": "Keyword matching with stop words, stemming and synonym rules. Returns the best-matching "
             "policy verbatim, or nothing if no title word matches. No LLM, no tokens.",
    "llm": "Claude receives the **whole** policy database (98 policies) in the prompt and answers from it.",
    "rag": f"A vector index (`{core.EMBED_MODEL}` embeddings) finds the {core.TOP_K} most similar policies; "
           "only those are sent to Claude.",
}


def run_all(question: str) -> dict:
    runnable = {k: fn for k, fn in core.ANSWER.items() if k == "rules" or core.api_key()}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {key: pool.submit(fn, question) for key, fn in runnable.items()}
    results = {k: RuntimeError("no ANTHROPIC_API_KEY is configured") for k in core.ANSWER if k not in runnable}
    for key, future in futures.items():
        try:
            results[key] = future.result()
        except (anthropic.AnthropicError, RuntimeError) as e:  # API errors, missing key, refusals
            results[key] = e
    return results


def show_result(key: str, r):
    st.markdown(f"#### {core.APPROACHES[key]}")
    st.caption(DESCRIPTIONS[key])
    if isinstance(r, Exception):
        st.error(f"Could not get an answer: {r}")
        return
    if r.found:
        st.success(r.answer)
    else:
        st.warning(r.answer)
    c1, c2 = st.columns(2)
    c1.metric("Response time", f"{r.seconds:.2f} s" if r.seconds >= 0.1 else f"{r.seconds * 1000:.1f} ms")
    c2.metric("Tokens", f"{r.tokens:,}", help=f"{r.input_tokens:,} input + {r.output_tokens:,} output")
    st.markdown("**Relevant policy**")
    if r.policies:
        for title in r.policies:
            p = core.by_title(title)
            st.markdown(f"📄 **{p['title']}** · {p['category']} · {p['department']}  \n> {p['policy_text']}")
    else:
        st.markdown("_None — the approach found no policy that covers the question._")
    if r.retrieved:
        label = "Top keyword scores" if key == "rules" else "Retrieved by vector search (cosine similarity)"
        with st.expander(label):
            st.dataframe(pd.DataFrame(r.retrieved, columns=["Policy", "Score"]), hide_index=True,
                         use_container_width=True)


def compare_tab():
    st.markdown("Ask a policy question. The same question is answered by all three approaches so you can "
                "compare the answer, the policy it relies on, the response time and the token use.")
    if not core.api_key():
        st.error("No `ANTHROPIC_API_KEY` is configured, so only the rules-based approach can run.")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, use_container_width=True):
            st.session_state["question"] = ex
    with st.form("ask"):
        question = st.text_input("Question", key="question",
                                 placeholder="e.g. Do I need approval before buying a $800 monitor?")
        submitted = st.form_submit_button("Compare", type="primary")
    if submitted and question.strip():
        with st.spinner("Asking all three approaches…"):
            st.session_state["last"] = (question.strip(), run_all(question.strip()))
    if "last" in st.session_state:
        question, results = st.session_state["last"]
        st.markdown(f"**Question:** {question}")
        cols = st.columns(3)
        for col, key in zip(cols, core.APPROACHES):
            with col, st.container(border=True):
                show_result(key, results[key])
        st.caption("Unsupported answers are measured on the evaluation set (see the Evaluation tab), where every "
                   "answer is checked against the policy database.")


# ---------------------------------------------------------------- evaluation

def evaluation_tab():
    if not RESULTS.exists():
        st.info("No evaluation results yet. Run `python evaluate.py` to create them.")
        return
    data = json.loads(RESULTS.read_text(encoding="utf8"))
    rows = pd.DataFrame(data["rows"])
    n = rows["question"].nunique()
    st.markdown(
        f"All three approaches answered the same **{n} questions**: answerable questions, questions where the "
        "relevant policy exists but lacks the detail asked for, and out-of-scope questions that no policy covers. "
        "An answer is **unsupported** when it states something the policy database does not support, or "
        "answers an out-of-scope question instead of declining. LLM answers were checked by a separate Claude "
        f"call that saw the whole database. Model: `{data['model']}`, run {data['run_at']}.")

    s = data["summary"]
    summary = pd.DataFrame({
        "Approach": [core.APPROACHES[k] for k in s],
        "Correct policy": [f"{v['correct_policy_pct']}%" for v in s.values()],
        "Unsupported answers": [f"{v['unsupported_pct']}% ({v['unsupported_count']})" for v in s.values()],
        "Median time": [f"{v['median_seconds']:.2f} s" for v in s.values()],
        "Mean tokens / question": [f"{v['mean_tokens']:,}" for v in s.values()],
        "Answerable: correct": [v["correct_answerable"] for v in s.values()],
        "Detail missing: unsupported": [v["unsupported_detail_missing"] for v in s.values()],
        "Out of scope: declined": [v["correct_out_of_scope"] for v in s.values()],
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)

    st.markdown("#### Every answer")
    approach = st.selectbox("Approach", list(core.APPROACHES.values()))
    key = next(k for k, v in core.APPROACHES.items() if v == approach)
    only_bad = st.checkbox("Show only wrong or unsupported answers")
    view = rows[rows["approach"] == key].copy()
    if only_bad:
        view = view[(~view["correct_policy"]) | (view["unsupported"])]
    view["policies"] = view["policies"].apply(", ".join)
    view["expected"] = view["expected"].apply(lambda e: ", ".join(e) or "(none — should decline)")
    st.dataframe(view[["question", "type", "answer", "policies", "expected", "correct_policy", "unsupported",
                       "judge_note", "seconds", "tokens"]],
                 hide_index=True, use_container_width=True,
                 column_config={"seconds": st.column_config.NumberColumn(format="%.2f")})


# ---------------------------------------------------------------- layout

st.title("📘 Company Policy Assistant")
st.caption("Answers employee questions about company policy and identifies the relevant policy.")

tab_compare, tab_eval, tab_writeup, tab_db = st.tabs(
    ["🔍 Compare live", "📊 Evaluation", "📝 Comparison & preferred approach", "📚 Policy database"])
with tab_compare:
    compare_tab()
with tab_eval:
    evaluation_tab()
with tab_writeup:
    if WRITEUP.exists():
        st.markdown(WRITEUP.read_text(encoding="utf8"))
    else:
        st.info("The write-up is added after the evaluation has been run.")
with tab_db:
    st.dataframe(pd.DataFrame(core.policies()), hide_index=True, use_container_width=True)
