# Company Policy Assistant

A bot that answers employee questions about company policy and identifies the relevant policy, built on a
database of 98 company policies (`data/company_policies.csv`).

## Three approaches

| Approach | How it works |
|---|---|
| Rules-based search | Keyword matching with stop words, light stemming and hand-written synonym rules; returns the best-matching policy verbatim, or nothing if no title word matches. No LLM. |
| LLM without vector index | Claude (`claude-opus-5-5`) receives the whole policy database in the prompt and answers from it. |
| LLM with vector index | A vector index of `BAAI/bge-small-en-v1.5` embeddings (via fastembed) retrieves the 3 most similar policies; only those are sent to Claude. |

All code for the approaches is in [`core.py`](core.py).

## Website

[`app.py`](app.py) is a Streamlit site with four tabs:

- **Compare live**: answers one question with all three approaches side by side, showing the answer, the
  relevant policy, the response time and the token use.
- **Evaluation**: results on 32 test questions (`data/eval_questions.json`), including how often each
  approach gives answers that the policy database does not support.
- **Comparison & preferred approach**: the two-paragraph write-up ([`WRITEUP.md`](WRITEUP.md)).
- **Policy database**: the 98 policies.

Viewers must enter the app password before anything is shown.

## Slack bot

[`slack_bot.py`](slack_bot.py) answers questions in Slack with the preferred approach (LLM with vector index)
and names the relevant policy. It uses Socket Mode, so it runs locally without a public URL.

1. Create an app at <https://api.slack.com/apps> → *Create New App* → *From a manifest*, and paste
   [`slack_manifest.yaml`](slack_manifest.yaml).
2. *Basic Information → App-Level Tokens*: create a token with the `connections:write` scope (`xapp-…`).
3. *Install App*: install to the workspace and copy the *Bot User OAuth Token* (`xoxb-…`).
4. Put both tokens and the Anthropic key in `.env` (see [`.env.example`](.env.example)), invite the bot to a
   channel (`/invite @Policy Assistant`) and run `python slack_bot.py`.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env          # then fill in the values
streamlit run app.py
python evaluate.py            # re-run the evaluation (calls the Claude API)
```

## Deploy (Streamlit Community Cloud)

Create an app from this repository with `app.py` as the main file, and add `ANTHROPIC_API_KEY` and
`APP_PASSWORD` under *Settings → Secrets*.

## Security

API keys, Slack tokens and the app password are read from `.env` locally or Streamlit Secrets when deployed;
both are git-ignored and never committed.
