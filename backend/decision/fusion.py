import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple

from backend.decision.models import MLSignal, IntelligenceSnapshot, FusionResult, TradeProposal
from backend.decision.policy import FusionPolicy
from backend.decision.exceptions import DecisionError
from backend.decision.telemetry import DecisionTelemetrySink, ConsoleDecisionTelemetrySink


class DecisionFusionEngine:
    """
    Deterministic Decision Fusion Engine for QuantForge.
    Combines fast-path ML signals with slow-path/asynchronous LLM Intelligence snapshots.
    Operates strictly deterministically with zero execution or trading authority.
    """

    def __init__(
        self,
        policy: FusionPolicy,
        telemetry_sink: Optional[DecisionTelemetrySink] = None,
    ):
        self.policy = policy
        self.telemetry_sink = telemetry_sink or ConsoleDecisionTelemetrySink()

    def normalize_direction(self, direction: str) -> str:
        """
        Normalize raw direction strings to a strict internal domain: BULLISH, BEARISH, NEUTRAL.
        Any invalid or unrecognized directional values fail closed (raise DecisionError).
        """
        if not direction:
            raise DecisionError("Direction value cannot be empty.")
        
        normalized = direction.strip().upper()
        if normalized in ("BULLISH", "BUY", "LONG"):
            return "BULLISH"
        elif normalized in ("BEARISH", "SELL", "SHORT"):
            return "BEARISH"
        elif normalized in ("NEUTRAL", "NONE", "HOLD", "NO_TRADE"):
            return "NEUTRAL"
        else:
            raise DecisionError(f"Invalid directional value detected: '{direction}'")

    def calculate_agreement(self, ml_dir: str, intel_dir: str) -> float:
        """
        Agreement score formula (v1):
        - Same direction (BULLISH/BULLISH, BEARISH/BEARISH, NEUTRAL/NEUTRAL): +1.0
        - Opposite directions (BULLISH/BEARISH, BEARISH/BULLISH): -1.0
        - Mixed neutrals (one is NEUTRAL, other is BULLISH/BEARISH): 0.0
        """
        norm_ml = self.normalize_direction(ml_dir)
        norm_intel = self.normalize_direction(intel_dir)

        if norm_ml == norm_intel:
            return 1.0
        elif (norm_ml == "BULLISH" and norm_intel == "BEARISH") or (
            norm_ml == "BEARISH" and norm_intel == "BULLISH"
        ):
            return -1.0
        else:
            return 0.0

    def fuse(
        self,
        ml_signal: MLSignal,
        intelligence_snapshot: Optional[IntelligenceSnapshot],
        market_context: Optional[Dict[str, Any]] = None,
        now: Optional[datetime] = None,
    ) -> Tuple[FusionResult, Optional[TradeProposal]]:
        """
        Performs deterministic fusion between MLSignal and IntelligenceSnapshot.
        Guarantees that LLM network or inference latency does not block execution.
        """
        start_counter = time.perf_counter()
        fusion_id = str(uuid.uuid4())
        check_time = now if now is not None else datetime.now(timezone.utc)
        if check_time.tzinfo is None:
            check_time = check_time.replace(tzinfo=timezone.utc)

        # 1. Base input validations
        if not ml_signal.symbol or not ml_signal.timeframe:
            raise DecisionError("MLSignal requires valid symbol and timeframe.")

        # 2. Check Intelligence availability and expiration (TTL)
        intel_available = False
        intel_expired = False
        intel_obj: Optional[IntelligenceSnapshot] = None
        intel_age_seconds: Optional[float] = None

        if intelligence_snapshot is not None:
            # Parse generating time of snapshot to calculate age
            try:
                gen_time = datetime.fromisoformat(intelligence_snapshot.generated_at)
                if gen_time.tzinfo is None:
                    gen_time = gen_time.replace(tzinfo=timezone.utc)
                intel_age_seconds = (check_time - gen_time).total_seconds()
            except Exception:
                intel_age_seconds = None

            # Verify expiration
            try:
                exp_ctime = datetime.fromisoformat(intelligence_snapshot.expires_at)
                if exp_ctime.tzinfo is None:
                    exp_ctime = exp_ctime.replace(tzinfo=timezone.utc)
                if check_time > exp_ctime:
                    intel_expired = True
                else:
                    intel_obj = intelligence_snapshot
                    intel_available = True
            except Exception:
                intel_expired = True

        # 3. Fast Path (ML-Only) check
        use_ml_only = not intel_available or intel_expired

        # Normalize direction domains
        try:
            ml_dir = self.normalize_direction(ml_signal.direction)
        except DecisionError as e:
            raise DecisionError(f"MLSignal direction normalization failure: {str(e)}") from e

        # Apply Safety Gate: Minimum ML Confidence
        if ml_signal.confidence < self.policy.minimum_ml_confidence:
            rejection = f"ML confidence ({ml_signal.confidence:.4f}) below minimum limit ({self.policy.minimum_ml_confidence:.4f})"
            res = self._build_rejected_result(
                fusion_id=fusion_id,
                ml_signal=ml_signal,
                intel_snapshot=intel_obj,
                intel_age=intel_age_seconds,
                reason=rejection,
                check_time=check_time,
            )
            elapsed_ms = (time.perf_counter() - start_counter) * 1000.0
            self.telemetry_sink.record(res, None, elapsed_ms, rejection)
            return res, None

        # Apply Safety Gate: ML Model Drift
        if ml_signal.drift_status.strip().lower() == "critical" and self.policy.reject_on_critical_drift:
            rejection = "Critical ML model drift detected; trade proposals rejected."
            res = self._build_rejected_result(
                fusion_id=fusion_id,
                ml_signal=ml_signal,
                intel_snapshot=intel_obj,
                intel_age=intel_age_seconds,
                reason=rejection,
                check_time=check_time,
            )
            elapsed_ms = (time.perf_counter() - start_counter) * 1000.0
            self.telemetry_sink.record(res, None, elapsed_ms, rejection)
            return res, None

        if use_ml_only:
            # Check if ML-only is allowed by policy
            if not self.policy.allow_ml_only:
                rejection = "Intelligence Context is unavailable or stale, and allow_ml_only is disabled."
                res = self._build_rejected_result(
                    fusion_id=fusion_id,
                    ml_signal=ml_signal,
                    intel_snapshot=None,
                    intel_age=None,
                    reason=rejection,
                    check_time=check_time,
                )
                elapsed_ms = (time.perf_counter() - start_counter) * 1000.0
                self.telemetry_sink.record(res, None, elapsed_ms, rejection)
                return res, None

            # ML-only fast path fusion logic implementation
            fusion_score = ml_signal.confidence
            final_dir = ml_dir
            final_confidence = ml_signal.confidence
            agreement_score = 0.0  # Undefined/Neutral in ML-only path

            # Pre- proposal threshold checks
            if final_confidence < self.policy.minimum_fusion_confidence:
                rejection = f"Fusion confidence ({final_confidence:.4f}) below minimum limit ({self.policy.minimum_fusion_confidence:.4f})"
                res = self._build_failed_gate_result(
                    fusion_id=fusion_id,
                    ml_signal=ml_signal,
                    final_dir=final_dir,
                    final_conf=final_confidence,
                    f_score=fusion_score,
                    ag_score=agreement_score,
                    intel_used=False,
                    intel_age=None,
                    rejection_reason=rejection,
                    check_time=check_time,
                )
                elapsed_ms = (time.perf_counter() - start_counter) * 1000.0
                self.telemetry_sink.record(res, None, elapsed_ms, rejection)
                return res, None

            # If resulting direction is neutral, we do not propose trades
            if final_dir == "NEUTRAL":
                rejection = "Neutral resulting direction"
                res = self._build_failed_gate_result(
                    fusion_id=fusion_id,
                    ml_signal=ml_signal,
                    final_dir=final_dir,
                    final_conf=final_confidence,
                    f_score=fusion_score,
                    ag_score=agreement_score,
                    intel_used=False,
                    intel_age=None,
                    rejection_reason=rejection,
                    check_time=check_time,
                )
                elapsed_ms = (time.perf_counter() - start_counter) * 1000.0
                self.telemetry_sink.record(res, None, elapsed_ms, rejection)
                return res, None

            # Create Proposal contracts
            proposal = TradeProposal(
                proposal_id=str(uuid.uuid4()),
                symbol=ml_signal.symbol,
                direction=final_dir,
                confidence=final_confidence,
                fusion_score=fusion_score,
                source_model_version=ml_signal.model_version,
                fusion_policy_version=self.policy.policy_version,
                reasoning_request_id=None,
                created_at=check_time.isoformat(),
                expires_at=(check_time + timedelta(seconds=self.policy.proposal_ttl_seconds)).isoformat(),
                risk_flags=[],
                metadata={"ml_only": True},
            )

            result = FusionResult(
                fusion_id=fusion_id,
                symbol=ml_signal.symbol,
                timeframe=ml_signal.timeframe,
                direction=final_dir,
                confidence=final_confidence,
                fusion_score=fusion_score,
                agreement_score=agreement_score,
                ml_contribution=self.policy.ml_weight,
                intelligence_contribution=self.policy.intelligence_weight,
                intelligence_used=False,
                intelligence_age_seconds=None,
                policy_version=self.policy.policy_version,
                source_model_version=ml_signal.model_version,
                reasoning_request_id=None,
                risk_flags=[],
                timestamp=check_time.isoformat(),
                metadata={"ml_only": True},
            )

            elapsed_ms = (time.perf_counter() - start_counter) * 1000.0
            self.telemetry_sink.record(result, proposal, elapsed_ms)
            return result, proposal

        # 4. Contextual Intelligence Path (Both ML & LLM Available)
        assert intel_obj is not None
        try:
            intel_dir = self.normalize_direction(intel_obj.directional_bias)
        except DecisionError as e:
            raise DecisionError(f"IntelligenceSnapshot direction normalization failure: {str(e)}") from e

        # Apply Safety Gate: Policy reject on intelligence risk flags
        matching_flags = [flag for flag in intel_obj.risk_flags if flag in self.policy.reject_on_intelligence_risk_flags]
        if matching_flags:
            rejection = f"Rejected due to critical intelligence risk flags: {matching_flags}"
            res = self._build_rejected_result(
                fusion_id=fusion_id,
                ml_signal=ml_signal,
                intel_snapshot=intel_obj,
                intel_age=intel_age_seconds,
                reason=rejection,
                check_time=check_time,
            )
            elapsed_ms = (time.perf_counter() - start_counter) * 1000.0
            self.telemetry_sink.record(res, None, elapsed_ms, rejection)
            return res, None

        agreement_score = self.calculate_agreement(ml_dir, intel_dir)

        # Apply Safety Gate: Minimum Agreement Score
        if agreement_score < self.policy.minimum_agreement_score:
            rejection = f"Agreement score ({agreement_score:.2f}) below minimum limit ({self.policy.minimum_agreement_score:.2f})"
            res = self._build_failed_gate_result(
                fusion_id=fusion_id,
                ml_signal=ml_signal,
                final_dir="NEUTRAL",
                final_conf=0.0,
                f_score=0.0,
                ag_score=agreement_score,
                intel_used=True,
                intel_age=intel_age_seconds,
                rejection_reason=rejection,
                snapshot=intel_obj,
                check_time=check_time,
            )
            elapsed_ms = (time.perf_counter() - start_counter) * 1000.0
            self.telemetry_sink.record(res, None, elapsed_ms, rejection)
            return res, None

        # Calculate Fusion Metrics
        ml_mapping = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}
        intel_mapping = {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}

        ml_val = ml_mapping[ml_dir]
        intel_val = intel_mapping[intel_dir]

        # Net directional math
        net_directional = (ml_val * ml_signal.confidence * self.policy.ml_weight) + (
            intel_val * intel_obj.confidence * self.policy.intelligence_weight
        )

        if net_directional > 0.0:
            final_dir = "BULLISH"
        elif net_directional < 0.0:
            final_dir = "BEARISH"
        else:
            final_dir = "NEUTRAL"

        # Confidence calculation (weighted average)
        final_confidence = (ml_signal.confidence * self.policy.ml_weight) + (
            intel_obj.confidence * self.policy.intelligence_weight
        )
        fusion_score = abs(net_directional)

        # Apply Safety Gate: Minimum Fusion Confidence
        if final_confidence < self.policy.minimum_fusion_confidence:
            rejection = f"Fusion confidence ({final_confidence:.4f}) below minimum limit ({self.policy.minimum_fusion_confidence:.4f})"
            res = self._build_failed_gate_result(
                fusion_id=fusion_id,
                ml_signal=ml_signal,
                final_dir=final_dir,
                final_conf=final_confidence,
                f_score=fusion_score,
                ag_score=agreement_score,
                intel_used=True,
                intel_age=intel_age_seconds,
                rejection_reason=rejection,
                snapshot=intel_obj,
                check_time=check_time,
            )
            elapsed_ms = (time.perf_counter() - start_counter) * 1000.0
            self.telemetry_sink.record(res, None, elapsed_ms, rejection)
            return res, None

        # If final direction is NEUTRAL, we cannot propose trades
        if final_dir == "NEUTRAL":
            rejection = "resulting direction is neutral"
            res = self._build_failed_gate_result(
                fusion_id=fusion_id,
                ml_signal=ml_signal,
                final_dir=final_dir,
                final_conf=final_confidence,
                f_score=fusion_score,
                ag_score=agreement_score,
                intel_used=True,
                intel_age=intel_age_seconds,
                rejection_reason=rejection,
                snapshot=intel_obj,
                check_time=check_time,
            )
            elapsed_ms = (time.perf_counter() - start_counter) * 1000.0
            self.telemetry_sink.record(res, None, elapsed_ms, rejection)
            return res, None

        # Create Proposal contracts
        proposal = TradeProposal(
            proposal_id=str(uuid.uuid4()),
            symbol=ml_signal.symbol,
            direction=final_dir,
            confidence=final_confidence,
            fusion_score=fusion_score,
            source_model_version=ml_signal.model_version,
            fusion_policy_version=self.policy.policy_version,
            reasoning_request_id=intel_obj.request_id,
            created_at=check_time.isoformat(),
            expires_at=(check_time + timedelta(seconds=self.policy.proposal_ttl_seconds)).isoformat(),
            risk_flags=list(intel_obj.risk_flags),
            metadata={"intel_snapshot_id": intel_obj.snapshot_id},
        )

        result = FusionResult(
            fusion_id=fusion_id,
            symbol=ml_signal.symbol,
            timeframe=ml_signal.timeframe,
            direction=final_dir,
            confidence=final_confidence,
            fusion_score=fusion_score,
            agreement_score=agreement_score,
            ml_contribution=self.policy.ml_weight,
            intelligence_contribution=self.policy.intelligence_weight,
            intelligence_used=True,
            intelligence_age_seconds=intel_age_seconds,
            policy_version=self.policy.policy_version,
            source_model_version=ml_signal.model_version,
            reasoning_request_id=intel_obj.request_id,
            risk_flags=list(intel_obj.risk_flags),
            timestamp=check_time.isoformat(),
            metadata={"intel_snapshot_id": intel_obj.snapshot_id},
        )

        elapsed_ms = (time.perf_counter() - start_counter) * 1000.0
        self.telemetry_sink.record(result, proposal, elapsed_ms)
        return result, proposal

    def _build_rejected_result(
        self,
        fusion_id: str,
        ml_signal: MLSignal,
        intel_snapshot: Optional[IntelligenceSnapshot],
        intel_age: Optional[float],
        reason: str,
        check_time: datetime,
    ) -> FusionResult:
        return FusionResult(
            fusion_id=fusion_id,
            symbol=ml_signal.symbol,
            timeframe=ml_signal.timeframe,
            direction="NEUTRAL",
            confidence=0.0,
            fusion_score=0.0,
            agreement_score=0.0,
            ml_contribution=self.policy.ml_weight,
            intelligence_contribution=self.policy.intelligence_weight,
            intelligence_used=intel_snapshot is not None,
            intelligence_age_seconds=intel_age,
            policy_version=self.policy.policy_version,
            source_model_version=ml_signal.model_version,
            reasoning_request_id=intel_snapshot.request_id if intel_snapshot else None,
            risk_flags=list(intel_snapshot.risk_flags) if intel_snapshot else [],
            timestamp=check_time.isoformat(),
            metadata={"rejection_reason": reason},
        )

    def _build_failed_gate_result(
        self,
        fusion_id: str,
        ml_signal: MLSignal,
        final_dir: str,
        final_conf: float,
        f_score: float,
        ag_score: float,
        intel_used: bool,
        intel_age: Optional[float],
        rejection_reason: str,
        snapshot: Optional[IntelligenceSnapshot] = None,
        check_time: Optional[datetime] = None,
    ) -> FusionResult:
        ctime = check_time or datetime.now(timezone.utc)
        return FusionResult(
            fusion_id=fusion_id,
            symbol=ml_signal.symbol,
            timeframe=ml_signal.timeframe,
            direction=final_dir,
            confidence=final_conf,
            fusion_score=f_score,
            agreement_score=ag_score,
            ml_contribution=self.policy.ml_weight,
            intelligence_contribution=self.policy.intelligence_weight,
            intelligence_used=intel_used,
            intelligence_age_seconds=intel_age,
            policy_version=self.policy.policy_version,
            source_model_version=ml_signal.model_version,
            reasoning_request_id=snapshot.request_id if snapshot else None,
            risk_flags=list(snapshot.risk_flags) if snapshot else [],
            timestamp=ctime.isoformat(),
            metadata={"rejection_reason": rejection_reason},
        )
