from typing import Dict, Any

PROMPT_TEMPLATES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "market_reasoning": {
        "v1": {
            "prompt_id": "market_reasoning",
            "version": "v1",
            "task_type": "reasoning",
            "system_template": (
                "You are an expert quantitative trading assistant. Your task is to analyze market indicators, "
                "predictions, and risk bounds to produce a structured market analysis. "
                "You must respond ONLY with a JSON object matching the requested schema. "
                "Do not include any other text, markdown wrappers, or external comments outside the JSON block. "
                "Under NO circumstances can you execute trades, issue orders, or interface with brokers. "
                "Any confidence score must be a float between 0.0 and 1.0."
            ),
            "user_template": (
                "For symbol {symbol} and timeframe {timeframe} at timestamp {timestamp}:\n"
                "Features parsed: {features}\n"
                "ML Preds: {ml_predictions}\n"
                "Market context: {market_context}\n"
                "Portfolio details: {portfolio_context}\n"
                "Risk context limits: {risk_context}\n"
                "Additional notes: {additional_context}\n"
                "Perform market analysis and emit the required JSON formatting."
            ),
            "response_schema": {
                "type": "object",
                "properties": {
                    "market_regime": {"type": "string"},
                    "directional_bias": {"type": "string"},
                    "confidence": {"type": "number"},
                    "risk_flags": {"type": "array", "items": {"type": "string"}},
                    "reasoning_summary": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "market_regime",
                    "directional_bias",
                    "confidence",
                    "risk_flags",
                    "reasoning_summary",
                    "evidence",
                ],
            },
        }
    }
}


def get_prompt_template(prompt_id: str, version: str) -> Dict[str, Any]:
    """
    Retrieve a versioned prompt template.
    Raises KeyError if prompt_id or version is not found.
    """
    if prompt_id not in PROMPT_TEMPLATES:
        raise KeyError(f"Prompt ID '{prompt_id}' is not registered.")
    if version not in PROMPT_TEMPLATES[prompt_id]:
        raise KeyError(f"Version '{version}' is not registered for prompt ID '{prompt_id}'.")
    return PROMPT_TEMPLATES[prompt_id][version]
