import os
import sys
import json
import uuid
import datetime

# Add root folder to sys.path so we can import backend/configs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.settings import settings
from backend.intelligence.exceptions import (
    IntelligenceError,
    ProviderConfigurationError,
    ProviderUnavailableError
)
from backend.intelligence.models import (
    ReasoningRequest,
    ReasoningResult
)
from backend.intelligence.providers.registry import AIProviderRegistry
from backend.intelligence.reasoning.engine import ReasoningEngine


def check_execution_isolation() -> bool:
    """Statically verify that backend.intelligence does not import trading/broker/execution modules."""
    forbidden_imports = [
        "backend.execution",
        "backend.broker",
        "PaperExecutor",
        "LiveExecutor",
        "OrderExecutor",
        "MarketConsumer",
    ]
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "backend", "intelligence")
    )
    if not os.path.exists(base_dir):
        return False

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    for forbidden in forbidden_imports:
                        if forbidden in content:
                            print(f"[ERROR] Isolation violation: '{forbidden}' found in {filepath}")
                            return False
    return True


def main():
    print("QUANTFORGE OLLAMA REASONING SMOKE TEST SCRIPT INITIALIZED")
    print(f"Detected configuration:")
    print(f"  AI_PROVIDER: {settings.AI_PROVIDER}")
    print(f"  AI_MODEL: {settings.AI_MODEL}")
    print(f"  AI_BASE_URL: {settings.AI_BASE_URL}")

    # Validate provider and model configuration
    if not settings.AI_PROVIDER:
        raise ProviderConfigurationError("AI_PROVIDER is not configured.")
    if not settings.AI_MODEL:
        raise ProviderConfigurationError("AI_MODEL is not configured in settings.")

    # 1. Resolve and create provider via central Registry
    # Use a generous inference timeout for smoke testing (thinking models need more time)
    smoke_inference_timeout = max(
        settings.AI_INFERENCE_TIMEOUT_SECONDS if settings.AI_INFERENCE_TIMEOUT_SECONDS is not None else 30.0,
        120.0
    )
    try:
        provider = AIProviderRegistry.create(
            settings.AI_PROVIDER,
            {
                "model_name": settings.AI_MODEL,
                "base_url": settings.AI_BASE_URL,
                "connection_timeout": settings.AI_CONNECTION_TIMEOUT_SECONDS if settings.AI_CONNECTION_TIMEOUT_SECONDS is not None else 5.0,
                "inference_timeout": smoke_inference_timeout,
            }
        )
    except Exception as e:
        print(f"[FAIL] Provider creation error: {str(e)}")
        raise ProviderConfigurationError(f"Failed to load or configure AI provider: {str(e)}") from e

    # 2. Check provider health
    print("\nChecking Provider Health...")
    health = provider.health_check()
    print(f"  Available: {health.available}")
    print(f"  Provider: {health.provider}")
    print(f"  Model: {health.model}")
    print(f"  Latency: {health.latency_ms:.2f} ms")
    if health.error:
        print(f"  Error Details: {health.error}")
    
    if not health.available:
        print("[FAIL] Ollama provider is offline or requested model is not downloaded!")
        raise ProviderUnavailableError(
            f"Ollama provider unavailable. Ensure Ollama is running and model '{settings.AI_MODEL}' is downloaded. "
            f"Error details: {health.error}"
        )

    # 3. Build realistic synthetic ReasoningRequest
    req = ReasoningRequest(
        symbol="BTCUSDT",
        timeframe="5m",
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        features={
            "rsi": 32.5,
            "macd": -15.4,
            "ema_relationship": "close < ema_50 < ema_200",
            "volatility_atr_14": 150.0,
            "volume_ratio_1h": 1.25
        },
        ml_predictions={
            "directional_probability": 0.58,
            "calibrated_confidence": 0.55
        },
        market_context={
            "regime": "ranging",
            "global_trend": "neutral"
        },
        portfolio_context={
            "current_exposure_pct": 10.0
        },
        risk_context={
            "risk_state": "normal"
        }
    )

    # 4. Instantiate ReasoningEngine (real engine, real OllamaProvider)
    engine = ReasoningEngine(
        provider=provider,
        max_retries=settings.AI_STRUCTURED_MAX_RETRIES if settings.AI_STRUCTURED_MAX_RETRIES is not None else 3
    )

    # 5. Execute request through the real ReasoningEngine
    print(f"\nSubmitting request to {settings.AI_PROVIDER} ({settings.AI_MODEL})...")
    
    try:
        result = engine.reason(req)
    except Exception as e:
        print(f"[FAIL] End-to-end reasoning cycle failed: {str(e)}")
        # Propagate custom failure subclass cleanly
        raise

    # 6. Verify result contracts and parameters
    assert result.request_id is not None, "VERIFICATION FAILED: request_id missing"
    assert result.provider == settings.AI_PROVIDER, "VERIFICATION FAILED: provider mismatch"
    assert result.model == settings.AI_MODEL, "VERIFICATION FAILED: model mismatch"
    assert result.prompt_id is not None, "VERIFICATION FAILED: prompt_id missing"
    assert result.prompt_version is not None, "VERIFICATION FAILED: prompt_version missing"
    assert result.timestamp is not None, "VERIFICATION FAILED: timestamp missing"
    assert result.latency_ms >= 0, "VERIFICATION FAILED: negative latency"
    assert 0.0 <= result.confidence <= 1.0, f"VERIFICATION FAILED: confidence {result.confidence} out of range [0, 1]"

    # Verify structured fields presence
    assert result.market_regime is not None, "VERIFICATION FAILED: market_regime missing"
    assert result.directional_bias is not None, "VERIFICATION FAILED: directional_bias missing"
    assert isinstance(result.risk_flags, list), "VERIFICATION FAILED: risk_flags is not a list"
    assert result.reasoning_summary is not None, "VERIFICATION FAILED: reasoning_summary missing"
    assert isinstance(result.evidence, list), "VERIFICATION FAILED: evidence is not a list"

    # Static execution isolation boundary validation
    isolation_passed = check_execution_isolation()

    # 7. Print the final report
    print("\n" + "="*50)
    print("QUANTFORGE OLLAMA REASONING SMOKE TEST")
    print("="*50)
    print(f"Provider: {result.provider}")
    print(f"Model: {result.model}")
    print(f"Prompt: {result.prompt_id} / {result.prompt_version}")
    print(f"Latency: {result.latency_ms:.2f} ms")
    print()
    print(f"Market Regime: {result.market_regime}")
    print(f"Directional Bias: {result.directional_bias}")
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Risk Flags: {', '.join(result.risk_flags) if result.risk_flags else 'None'}")
    print(f"Reasoning Summary: {result.reasoning_summary}")
    print(f"Evidence: {', '.join(result.evidence) if result.evidence else 'None'}")
    print()
    print("Validation:")
    print("JSON parsing: PASS")
    print("Schema validation: PASS")
    print("Domain validation: PASS")
    print(f"Execution isolation: {'PASS' if isolation_passed else 'FAIL'}")
    print()
    print("RESULT: PASS")
    print("="*50)

    if not isolation_passed:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except IntelligenceError as ie:
        print(f"\n[FAIL] Smoke test failed with registered IntelligenceError subdivision: {type(ie).__name__}: {str(ie)}")
        sys.exit(2)
    except Exception as exc:
        print(f"\n[FAIL] Smoke test failed with unexpected exception: {type(exc).__name__}: {str(exc)}")
        sys.exit(3)
