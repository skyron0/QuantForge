from typing import Dict, Any

PROMPT_TEMPLATES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "market_reasoning": {
        "v1": {
            "prompt_id": "market_reasoning",
            "version": "v1",
            "task_type": "reasoning",
            "system_template": (
                "You are an expert quantitative trading analyst. "
                "You MUST respond with ONLY a single valid JSON object. "
                "Do NOT include any text, explanations, markdown, or code fences outside the JSON. "
                "Under NO circumstances can you execute trades, issue orders, or interface with brokers. "
                "You must output exactly this JSON structure with these exact field names:\n"
                '{{\n'
                '  "market_regime": "<trending_up|trending_down|ranging|volatile|breakout>",\n'
                '  "directional_bias": "<bullish|bearish|neutral>",\n'
                '  "confidence": <float between 0.0 and 1.0>,\n'
                '  "risk_flags": ["<flag1>", "<flag2>"],\n'
                '  "reasoning_summary": "<one paragraph analysis>",\n'
                '  "evidence": ["<evidence1>", "<evidence2>"]\n'
                '}}\n'
                "All six fields are REQUIRED. confidence must be a number between 0.0 and 1.0. "
                "risk_flags and evidence must be arrays of strings."
            ),
            "user_template": (
                "Analyze {symbol} on {timeframe} timeframe at {timestamp}.\n"
                "Technical indicators: {features}\n"
                "ML predictions: {ml_predictions}\n"
                "Market context: {market_context}\n"
                "Portfolio: {portfolio_context}\n"
                "Risk: {risk_context}\n"
                "Additional: {additional_context}\n"
                "Respond with ONLY the JSON object containing: market_regime, directional_bias, "
                "confidence, risk_flags, reasoning_summary, evidence."
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
