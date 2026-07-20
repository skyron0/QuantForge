# ADR-010: QuantForge Intelligence Architecture

## Title

QuantForge Intelligence & Provider Abstraction Architecture

## Status

Accepted

## Context

As QuantForge expands into Machine Learning (ML), Deep Learning (DL), and Large Language Model (LLM) decision layers, we require a architecture that enables:

1. **Multi-Provider Support**: Seamless swapping between local models (Ollama, llama.cpp, vLLMs, ONNX) and remote APIs (OpenAI, Anthropic) without rewriting trading strategies.
2. **Reproducibility**: Running identical model predictions inside Walk Forward Analysis backtests as we do in live production.
3. **Execution Safety**: Preventing LLM reasoning models from executing trades directly on exchanges, ensuring all orders pass through risk managers.

## Decision

1. **Unified AI Contracts**: Define standardized, provider-agnostic domain objects:
   - `PredictionRequest`: Wraps standardized features and timeframe queries.
   - `PredictionResponse`: Captures model outcomes and confidence scores.
   - `TradeProposal`: Represents the logical trading hypothesis (BUY/SELL/HOLD, targets, reasoning).
   - `TradeDecision`: The finalized execution directive approved/modified by risk boundaries.
2. **Provider Abstraction Interface**: Introduce a `BaseProvider` contract class. Provider adapters (e.g. `OllamaProvider`, `ONNXProvider`) must map proprietary formats to our Unified AI Contracts.
3. **Execution Firewall**: No intelligence component or reasoning agent may connect to exchange endpoints. All decisions are output as `TradeProposal` data structures delegates to the existing execution layer.
4. **Validation Progression**: Any AI strategy must be evaluated sequentially through Backtesting, Walk Forward Analysis, Parameter Optimization, and Shadow Mode before managing live assets.

## Consequences

- **Decoupled Strategies**: Strategies write business logic against predictions, remaining agnostic of the model type (XGBoost, Transformer, LLM) or host runtime.
- **Provider Agnostic**: Migrating from OpenAI to local Ollama or vLLM deployments requires zero changes to strategy definitions.
- **Strict Safety**: The trading engine retains absolute veto power over AI proposals via risk management configurations.
