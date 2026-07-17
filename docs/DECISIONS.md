\# QuantForge Architectural Decisions



This document records important architectural decisions.



The purpose is to preserve engineering intent.



\---



\# ADR-001



Title



Single Trading Pipeline



Status



Accepted



Reason



All trading decisions must pass through a single execution path.



Separate implementations inevitably diverge over time.



Decision



Both Live Trading and Backtesting must reuse the same pipeline.



Implementation



Historical Candle



↓



Consumer.process\_candle()



↓



Decision Engine



↓



Signal Validator



↓



Risk Manager



↓



Executor



\---



\# ADR-002



Title



Consumer Owns the Pipeline



Status



Accepted



Reason



Execution order must exist in exactly one place.



Decision



Consumer is responsible only for orchestration.



It contains no business logic.



\---



\# ADR-003



Title



Repositories Never Contain Business Logic



Status



Accepted



Reason



Repositories should only communicate with the database.



Decision



Repositories perform CRUD operations only.



\---



\# ADR-004



Title



Decision Engine Never Executes Orders



Status



Accepted



Reason



Decision making and execution are independent responsibilities.



Decision



Decision Engine returns BUY / SELL / HOLD only.



Execution belongs to the Executor.



\---



\# ADR-005



Title



Risk Manager Is Mandatory



Status



Accepted



Reason



Every position must pass through risk validation.



Decision



No order bypasses Risk Manager.



\---



\# ADR-006



Title



AI Replaces Only Decision Logic



Status



Accepted



Reason



The execution pipeline has already been validated.



Replacing only the decision layer minimizes risk.



Decision



Rule-Based



↓



Decision Engine



↓



Executor



AI



↓



AI Model



↓



Executor



\---



\# ADR-007



Title



Backtesting Must Match Production



Status



Accepted



Reason



Different execution paths produce misleading results.



Decision



Backtesting reuses production code.



No duplicated trading logic.



\---



\# ADR-008



Title



Documentation Is Part of the Codebase



Status



Accepted



Reason



AI agents and future contributors require architectural context.



Decision



Whenever architecture changes, update:



\- PROJECT\_STATE.md

\- ARCHITECTURE.md

\- DECISIONS.md

\- CHANGELOG.md



\---



\# Future ADRs



Every important architectural decision must be documented here before implementation.

