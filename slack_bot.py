"""Slack bot that answers policy questions and names the relevant policy.

Runs locally with Socket Mode, so no public URL is needed. Mention the bot in a channel
("@Policy Assistant how many vacation days do I get?") or send it a direct message.

Needs in .env (git-ignored): SLACK_BOT_TOKEN (xoxb-...), SLACK_APP_TOKEN (xapp-...), ANTHROPIC_API_KEY.
usage: python slack_bot.py
"""
import logging
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import os  # noqa: E402

from slack_bolt import App  # noqa: E402
from slack_bolt.adapter.socket_mode import SocketModeHandler  # noqa: E402

import core  # noqa: E402

logging.basicConfig(level=logging.INFO)
app = App(token=os.environ["SLACK_BOT_TOKEN"])


def reply_text(question: str) -> str:
    r = core.answer_rag(question)  # the preferred approach (see WRITEUP.md)
    lines = [r.answer]
    if r.policies:
        for title in r.policies:
            p = core.by_title(title)
            lines.append(f"\n:page_facing_up: *Relevant policy: {p['title']}* ({p['category']})\n>{p['policy_text']}")
    else:
        lines.append("\n:grey_question: _No relevant policy found in the policy database._")
    lines.append(f"\n_Answered from the policy database in {r.seconds:.1f} s · {r.tokens:,} tokens_")
    return "\n".join(lines)


def answer(question: str, say, thread_ts: str):
    question = re.sub(r"<@[A-Z0-9]+>", "", question).strip()  # drop the @mention
    if not question:
        say(text="Ask me a question about company policy, e.g. _How many vacation days do I get?_",
            thread_ts=thread_ts)
        return
    try:
        text = reply_text(question)
    except Exception as e:  # keep the bot running whatever goes wrong with one question
        logging.exception("answer failed")
        text = f":warning: Sorry, I could not answer that right now ({type(e).__name__})."
    say(text=text, thread_ts=thread_ts)


@app.event("app_mention")
def on_mention(event, say):
    answer(event.get("text", ""), say, event.get("thread_ts") or event["ts"])


@app.event("message")
def on_direct_message(event, say):
    if event.get("channel_type") == "im" and not event.get("bot_id") and not event.get("subtype"):
        answer(event.get("text", ""), say, event.get("thread_ts") or event["ts"])


if __name__ == "__main__":
    core.warm_up()
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
