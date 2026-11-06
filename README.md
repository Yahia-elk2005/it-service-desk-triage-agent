# IT Service Desk Triage & Routing Agent

Turns a plain-text IT problem into a structured ticket, like a small ServiceNow/ITSM triage desk.
Built with a free LLM API (Groq by default) and a Gradio web UI.

## What it does

1. **Receive request** - a user describes a tech issue in plain text, for example
   "My printer won't connect and I can't print payroll".
2. **Classify and prioritize** - the agent sets the priority (Low, Medium, Critical)
   and a department (Network, Hardware, Software, Security, Access).
3. **Route and draft** - it returns a JSON payload with priority, tags, assigned queue,
   response target and numbered draft fix steps.

Extra features:
- A check step that blocks unrelated requests and prompt-injection attempts.
- Users, each with their own saved tickets (SQLite).
- Per-user memory: ask "What was my last ticket?" or "How many Critical tickets do I have?".
  The model only builds a plan; the code runs fixed, read-only SQL.
- Live progress strip, priority bar, ticket card, JSON tab and history table.

## Structure (7 blocks in `service_desk_agent.py`)

1. Config and LLM client 2. Data models 3. Guardrail agent 4. Classifier agent
5. Router and resolution agent 6. Orchestrator, memory and users 7. UI

## Run it

```bash
python -m venv venv
venv\Scripts\activate        # Windows (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env         # Windows: copy .env.example .env
# open .env and paste your free API key
python service_desk_agent.py
```

Then open http://127.0.0.1:7860.

## Free providers

| Provider | Get a key | `.env` |
|---|---|---|
| Groq (default) | https://console.groq.com/keys | `GROQ_API_KEY` |
| Google Gemini | https://aistudio.google.com/apikey | `LLM_PROVIDER=gemini`, `GEMINI_API_KEY` |
| OpenRouter | https://openrouter.ai/keys | `LLM_PROVIDER=openrouter`, `OPENROUTER_API_KEY` |
| Ollama (local) | https://ollama.com | `LLM_PROVIDER=ollama` |

## Notes

- There is no login: anyone using the page can pick any user. Add authentication before real use.
- Tickets are stored in `tickets.sqlite` next to the script (ignored by git).
- Never commit your `.env` file.
