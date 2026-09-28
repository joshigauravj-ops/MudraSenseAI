# 🏛️ MudraSense AI

[![Python Version](https://shields.io)](https://python.org)
[![Framework](https://shields.io)](https://github.com)
[![Market](https://shields.io)]()
[![License](https://shields.io)]()

**MudraSense AI** is a production-grade, stateful multi-agent orchestrator built to track open trade positions, calculate real-time portfolio metrics, and analyze market sentiment for the **Indian Equities Market (NSE/BSE)**—completely free of cost.

By decoupling deterministic execution layers from LLM reasoning pools, the engine eliminates financial hallucinations while delivering autonomous risk management. The retrieval layer now uses live market context, ticker-scoped headlines, and a local document store as the default fallback; an optional FAISS-backed vector store can be enabled when a richer document index is needed.

**Development credit:** Gaurav Joshi ([GitHub](https://github.com/joshigauravj-ops))

Read the full system design, data flow, retrieval architecture, and guardrails in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 🧠 Core System Workflow

[ Input: Positions CSV ] ➔ [ Typed Validation ] ➔ [ yfinance + RSS Retrieval ]
➔ [ Local Document Store / Optional FAISS Vector Index ]
➔ [ Pure Python PnL Math ] ➔ [ LangGraph Agents ] ➔ [ Risk Guardrail ]

1. **Telemetry Retrieval:** Pulls data for NSE tickers via `yfinance` and parses current context from Google News RSS.
2. **Grounded Document Retrieval:** Queries the ticker-specific local fallback store and can optionally use a FAISS-backed vector index when configured.
3. **Deterministic Computation:** A pure Python mathematical core calculates open position profits, losses, and percentages in **INR (₹)**.
4. **Agentic Synthesis:** LangGraph routes data to free LLM nodes (via Groq/Ollama) to extract qualitative market context grounded in retrieved headlines and document excerpts.
5. **Autonomous Guardrails:** Instantly alerts the user via webhooks or terminal banners if portfolio stop-losses are breached.

---

## 🚀 Local Deployment Setup
[ Input: Positions CSV ] ➔ [ Typed Validation ] ➔ [ yfinance + RSS Retrieval ]
➔ [ Pure Python PnL Math ] ➔ [ LangGraph Agents ] ➔ [ Risk Guardrail ]
The project folder may live anywhere, including your user drive or a project
root. Ollama is a separate Windows application and should not be moved into the
project folder. The dashboard connects to Ollama at `http://127.0.0.1:11434`.

### 1. Initialize and Isolate Environment
From the project folder in PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If the virtual environment does not exist yet:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The current retrieval pipeline is a hybrid live RAG flow: fresh headlines are
retrieved for the ticker, and the local document store provides a deterministic
fallback for relevant company filings and excerpts. An optional FAISS-backed
vector index can be enabled for richer historical document retrieval when
configured in the environment.

### 2. Install and Start Ollama (Windows)
Ollama must be installed separately from this Python project. In PowerShell:

```powershell
winget install Ollama.Ollama
```

Close and reopen PowerShell after installation, then verify the command:

```powershell
ollama --version
```

Start the local Ollama service in a separate terminal:

```powershell
ollama serve
```

Keep that terminal running. In another terminal, download the configured model:

```powershell
ollama pull llama3.2:3b
```

The application uses `llama3.2:3b` by default in `.env.example`. If `ollama`
is not recognized, reopen PowerShell or add the Ollama installation directory
to your Windows user `PATH`; do not copy the Ollama installation into this
project.

### 3. Install Pinned Dependencies
Create a `requirements.txt` file and populate it:
```text
langgraph>=0.0.10
groq>=0.5.0
yfinance>=0.2.40
pydantic>=2.7.0
requests>=2.31.0
beautifulsoup4>=4.12.0
certifi>=2024.05.10
python-dotenv>=1.0.1
```
Install them inside the virtual workspace:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Copy the committed template to a private `.env` file:

```powershell
Copy-Item .env.example .env
```

Set `GROQ_API_KEY` if using Groq, or keep `MUDRASENSE_PROVIDER=ollama` for local
inference. `.env` is git-ignored and must never contain committed credentials.
The template also contains model names, Ollama host, input path, network timeouts,
supervisor thresholds, and optional retrieval settings such as the local fallback,
`MUDRASENSE_READ_BACKEND`, and FAISS index path when enabled.

To use the optional vector back end, set the provider to `faiss` and provide a
valid `MUDRASENSE_FAISS_INDEX_PATH` when an index file has been created. If that
backend is unavailable, the project automatically falls back to the local
retrieval store without failing the workflow.

### 5. Create Your Local Positions CSV
The repository includes only a template. Copy it to a local, ignored file and
edit it with your own open positions:

```powershell
Copy-Item examples/open_positions.template.csv positions.csv
```

Required CSV columns are `ticker`, `entry_price`, `quantity`, and `trade date` (or
`transaction_date`). The template also demonstrates `target price`, `target profit %`,
`target holding`, `action (B/S)`, and `commission+STT`. Tickers may
be entered as `RELIANCE` or `RELIANCE.NS`; the loader
normalizes them to the NSE `.NS` format. User CSV files matching `positions*.csv` are
ignored by git.

Multiple rows may use the same ticker. Each row is treated as a separate open
lot, so different entry prices, quantities, dates, targets, and PnL values are
preserved independently. Live market data is fetched once per ticker and then
applied to each matching lot.

Columns beginning with `target` are user-owned inputs. The agentic workflow never
changes `target price`, `target profit`, or `target holding`. The user-owned trade
identity and execution fields (`ticker`, `entry_price`, `quantity`, `trade date`,
`action (B/S)`, and `commission+STT`) are also preserved.

The CSV stores stable user inputs and qualitative agent outputs only:
`technical sentiment`, `breaking news analysis`, and `risk tier`. Volatile market
snapshots (`current price`, `days high`, `days low`, `net change %`, `market
session`) and deterministic `pnl`/`pnl %` remain in runtime state and are not
written on every refresh. Before any successful CSV replacement, the previous
file is copied to a sibling backup named `<filename>.csv.backup`. These backups
are ignored by git.

For a buy (`B`), PnL rises when the live price is above entry. For a sell (`S`),
PnL rises when the live price is below entry. `commission+STT` is deducted by the
Python calculation layer before writing the authoritative `pnl` and `pnl %` values
back to the CSV.

### 6. Run the Full Workflow
```bash
python main.py
```

To fetch and print a standalone price/PnL report without running agents or
rewriting the CSV:

```bash
python main.py --report
```

The default input path comes from `MUDRASENSE_INPUT` in `.env` and is
`positions.csv` unless overridden. A command-line `--input` still takes precedence.

Startup validates every row before fetching market data. Once valid, it runs the
market refresh, news and technical agents, and supervisor guardrail. Ollama is the
default local inference backend; use Groq with:

```bash
python main.py --provider groq
```

### 7. Open the UI
```bash
python main.py --ui
```

The dashboard asks for the path to your local CSV and displays the full current
market snapshot, deterministic PnL, agent summaries, and whether each snapshot
is `Live Intraday` or `Post-Market Close`. Only stable edits and qualitative
agent results are written back to the CSV.

Market requests retry three times by default, and PnL calculation retries twice.
Override these values with `MUDRASENSE_MARKET_RETRY_ATTEMPTS`,
`MUDRASENSE_PNL_RETRY_ATTEMPTS`, and `MUDRASENSE_RETRY_DELAY_SECONDS`.

Set `stop loss` in the input CSV to enable a deterministic stop-loss trigger.
For buys, targets trigger when price rises to the target and stop-losses trigger
when price falls to the stop. Sell positions use inverse comparisons. A target
changes the runtime risk tier to `Low`; a stop-loss changes it to `High`. The
dashboard and terminal report show the trigger, but no order is placed
automatically.

