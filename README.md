# 🏛️ MudraSense AI

[![Python Version](https://shields.io)](https://python.org)
[![Framework](https://shields.io)](https://github.com)
[![Market](https://shields.io)]()
[![License](https://shields.io)]()

**MudraSense AI** is a production-grade, stateful multi-agent orchestrator built to track open trade positions, calculate real-time portfolio metrics, and analyze market sentiment for the **Indian Equities Market (NSE/BSE)**—completely free of cost. 

By decoupling deterministic execution layers from LLM reasoning pools, the engine eliminates financial hallucinations while delivering autonomous risk management.

**Development credit:** Gaurav Joshi ([GitHub](https://github.com/joshigauravj-ops))

Read the full system design, data flow, RAG explanation, guardrails, and interview
showcase narrative in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 🧠 Core System Workflow

[ Input: Positions CSV ] ➔ [ Typed Validation ] ➔ [ yfinance + RSS Retrieval ]
➔ [ Pure Python PnL Math ] ➔ [ LangGraph Agents ] ➔ [ Risk Guardrail ]

1. **Telemetry Retrieval:** Pulls data for NSE tickers via `yfinance` and parses current context from Google News RSS.
2. **Deterministic Computation:** A pure Python mathematical core calculates open position profits, losses, and percentages in **INR (₹)**.
3. **Agentic Synthesis:** LangGraph routes data to free LLM nodes (via Groq/Ollama) to extract qualitative market context.
4. **Autonomous Guardrails:** Instantly alerts the user via webhooks or terminal banners if portfolio stop-losses are breached.

---

## 🚀 Local Deployment Setup
[ Input: Positions CSV ] ➔ [ Typed Validation ] ➔ [ yfinance + RSS Retrieval ]
➔ [ Pure Python PnL Math ] ➔ [ LangGraph Agents ] ➔ [ Risk Guardrail ]
### 1. Initialize and Isolate Environment
```bash
2. **Deterministic Computation:** A pure Python mathematical core calculates open position profits, losses, and percentages in **INR (₹)**.
python3 -m venv venv
4. **Autonomous Guardrails:** Instantly alerts the user via webhooks or terminal banners if portfolio stop-losses are breached.

The current RSS-based retrieval is a lightweight live RAG pipeline: fresh headlines
are retrieved for the ticker and supplied to the news agent as grounded context.
Historical filings and vector search are documented as a future extension in
[ARCHITECTURE.md](ARCHITECTURE.md).
```

### 2. Install Pinned Dependencies
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

### 3. Configure Environment
Copy the committed template to a private `.env` file:

```powershell
Copy-Item .env.example .env
```

Set `GROQ_API_KEY` if using Groq, or keep `MUDRASENSE_PROVIDER=ollama` for local
inference. `.env` is git-ignored and must never contain committed credentials.
The template also contains model names, Ollama host, input path, network timeouts,
and supervisor thresholds.

### 4. Create Your Local Positions CSV
The repository includes only a template. Copy it to a local, ignored file and
edit it with your own open positions:

```powershell
Copy-Item examples/open_positions.template.csv positions.csv
```

Required CSV columns are `ticker`, `entry_price`, `quantity`, and `trade date` (or
`transaction_date`). The template also demonstrates `current price`, `target price`,
`target profit`, `target holding`, `strike price`, `action (B/S)`, `commission+STT`,
and `pnl`. Tickers may be entered as `RELIANCE` or `RELIANCE.NS`; the loader
normalizes them to the NSE `.NS` format. User CSV files matching `positions*.csv` are
ignored by git.

Multiple rows may use the same ticker. Each row is treated as a separate open
lot, so different entry prices, quantities, dates, targets, and PnL values are
preserved independently. Live market data is fetched once per ticker and then
applied to each matching lot.

Columns beginning with `target` are user-owned inputs. The agentic workflow never
changes `target price`, `target profit`, or `target holding`. The user-owned trade
identity and execution fields (`ticker`, `entry_price`, `quantity`, `trade date`,
`action (B/S)`, `strike price`, and `commission+STT`) are also preserved.

The workflow updates these agent-owned columns in the same CSV after a successful
run: `current price`, `days high`, `days low`, `net change %`, `market session`,
`pnl`, `pnl %`, `technical sentiment`, `breaking news analysis`, and `risk tier`.
Before any successful CSV replacement, the previous file is copied to a sibling
backup named `<filename>.csv.backup`. These backups are ignored by git.

For a buy (`B`), PnL rises when the live price is above entry. For a sell (`S`),
PnL rises when the live price is below entry. `commission+STT` is deducted by the
Python calculation layer before writing the authoritative `pnl` and `pnl %` values
back to the CSV.

### 5. Run the Full Workflow
```bash
python main.py
```

The default input path comes from `MUDRASENSE_INPUT` in `.env` and is
`positions.csv` unless overridden. A command-line `--input` still takes precedence.

Startup validates every row before fetching market data. Once valid, it runs the
market refresh, news and technical agents, and supervisor guardrail. Ollama is the
default local inference backend; use Groq with:

```bash
python main.py --provider groq
```

### 6. Open the UI
```bash
python main.py --ui
```

The dashboard asks for the path to your local CSV and displays current price,
deterministic PnL, and whether each snapshot is `Live Intraday` or `Post-Market Close`.
