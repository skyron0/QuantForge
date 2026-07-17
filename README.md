# QuantForge

A modular, AI-ready algorithmic trading platform designed for research, backtesting, and live trading.

QuantForge is built around a single reusable trading pipeline, ensuring that backtesting and live trading share the same execution logic. This minimizes inconsistencies and simplifies long-term maintenance.

---

# Features

## Market Data

- Real-time WebSocket data collection
- Tick processing
- OHLCV candle aggregation
- Multi-symbol support

## Trading Pipeline

- Technical Indicator Engine
- Feature Engineering
- Decision Engine
- Signal Validation
- Risk Management
- Paper Trading Execution

## Data Storage

- PostgreSQL
- SQLAlchemy ORM
- Repository Pattern

## Monitoring

- Portfolio tracking
- Trade history
- Logging
- Dashboard

## AI Ready

- Feature generation
- Dataset export (planned)
- Local AI integration (planned)
- Ollama support (planned)

---

# Architecture

```
Market Data
     │
Collector
     │
Aggregator
     │
Consumer
     │
Indicator Engine
     │
Feature Engine
     │
Decision Engine
     │
Signal Validator
     │
Risk Manager
     │
Executor
     │
Repositories
     │
Database
```

Backtesting uses the same execution pipeline as live trading.

No duplicated trading logic.

---

# Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python |
| Database | PostgreSQL |
| Cache | Redis |
| ORM | SQLAlchemy |
| Containers | Docker |
| Dashboard | Streamlit |
| AI | Ollama (Planned) |

---

# Project Structure

```
QuantForge/

├── backend/
│   ├── collector/
│   ├── aggregator/
│   ├── consumer/
│   ├── indicators/
│   ├── features/
│   ├── decision/
│   ├── validation/
│   ├── risk/
│   ├── execution/
│   ├── repositories/
│   ├── models/
│   └── services/
│
├── dashboard/
├── docs/
├── tests/
├── docker/
│
├── AGENTS.md
├── README.md
├── CHANGELOG.md
└── CONTRIBUTING.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/skyron0/QuantForge.git
cd QuantForge
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Configuration

Create a `.env` file and configure:

```text
POSTGRES_HOST=
POSTGRES_PORT=
POSTGRES_DB=
POSTGRES_USER=
POSTGRES_PASSWORD=

REDIS_HOST=
REDIS_PORT=

MT5_LOGIN=
MT5_PASSWORD=
MT5_SERVER=

TELEGRAM_TOKEN=
TELEGRAM_CHAT_ID=
```

---

# Running

Start infrastructure:

```bash
docker compose up -d
```

Run the application:

```bash
python main.py
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

---

# Backtesting

Backtesting is currently under development.

The design goal is to replay historical candles through the same execution pipeline used for live trading.

```
Historical Data

↓

Consumer

↓

Decision Engine

↓

Risk Manager

↓

Executor

↓

Performance Metrics
```

---

# Roadmap

Current development roadmap:

- ✅ Infrastructure
- ✅ Trading Pipeline
- ✅ Paper Trading
- 🔄 Backtesting Engine
- ⏳ Performance Analytics
- ⏳ Dataset Generation
- ⏳ AI Decision Engine
- ⏳ Live Trading
- ⏳ Web Platform

Detailed planning is available in:

```
docs/ROADMAP.md
```

---

# Documentation

Project documentation is located in the `docs` directory.

- PROJECT_STATE.md
- ARCHITECTURE.md
- MODULES.md
- DECISIONS.md
- CODING_RULES.md
- DEVELOPMENT_WORKFLOW.md
- ROADMAP.md

---

# Contributing

Please read:

```
CONTRIBUTING.md
```

before submitting changes.

---

# License

This project is currently distributed without an open-source license.

All rights reserved unless stated otherwise.