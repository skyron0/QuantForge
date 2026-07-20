\# QuantForge - AI Agent Instructions

\## Mission

QuantForge is a production-oriented algorithmic trading platform.

Your responsibility is to improve the project without breaking the existing architecture.

Always preserve stability over speed.

\---

\# Development Principles

\- Never rewrite working code without a reason.

\- Never refactor unrelated modules.

\- Never rename folders unless explicitly requested.

\- Never duplicate business logic.

\- Reuse existing components whenever possible.

\- Keep the architecture modular.

\- Every implementation must leave the repository in a working state.

\---

\# Working Rules

For every task:

1\. Understand the current architecture.

2\. Explain the implementation plan.

3\. Modify only the required files.

4\. Verify consistency.

5\. Explain every modification.

\---

\# File Modification Rules

Maximum modified files per task:

2

Exceptions:

\- Dependency injection

\- Database migrations

\- Configuration updates

\---

\# Code Style

\- Clean Architecture

\- SOLID

\- DRY

\- Explicit names

\- Small functions

\- No dead code

\- No unused imports

\- Type hints whenever practical

\---

\# Testing Rules

Never finish after generating code.

Always verify:

\- imports

\- syntax

\- runtime consistency

If tests exist, run them.

\---

\# Architecture Rules

Trading pipeline is the single source of truth.

Collector

↓

Aggregator

↓

Consumer

↓

Indicators

↓

Feature Engine

↓

Decision Engine

↓

Signal Validator

↓

Risk Manager

↓

Executor

↓

Repositories

↓

Database

Never create an alternative execution pipeline.

Backtesting must reuse the same execution path.

\---

\# Documentation

Whenever architecture changes:

Update:

\- PROJECT_STATE.md

\- ARCHITECTURE.md

\- CHANGELOG.md

\---

\# Git Workflow

One feature

↓

Test

↓

Commit

↓

Push

Never combine multiple independent features into one commit.

\---

\# If uncertain

Stop.

Explain the uncertainty.

Do not guess.

No feature may exceed one sprint.

If implementation becomes larger than expected:

Stop.

Split the work.

Ask for approval.

AI Philosophy

QuantForge does not treat Large Language Models as trading engines.

Instead, AI is a collection of specialized intelligence modules.

Every AI module has a clear responsibility.

Trading decisions are evidence-driven and evaluated through deterministic research pipelines before reaching live execution.
