import os
import hashlib
import logging
from enum import Enum
from typing import Optional

from backend.training.lifecycle import ModelStatus
from backend.inference.exceptions import ArtifactIntegrityError

logger = logging.getLogger(__name__)


class IntegrityState(str, Enum):
    VERIFIED = "VERIFIED"
    UNVERIFIED_LEGACY = "UNVERIFIED_LEGACY"
    FAILED = "FAILED"


class IntegrityPolicy(str, Enum):
    STRICT = "STRICT"
    PRODUCTION_STRICT = "PRODUCTION_STRICT"
    LENIENT = "LENIENT"


def calculate_sha256(filepath: str) -> str:
    """Computes the SHA-256 hash of a file in chunks."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


class ArtifactIntegrityVerifier:
    def __init__(self, default_policy: IntegrityPolicy = IntegrityPolicy.PRODUCTION_STRICT):
        self.default_policy = default_policy

    def verify(
        self,
        filepath: str,
        registered_sha256: Optional[str],
        model_status: ModelStatus,
        policy: Optional[IntegrityPolicy] = None,
    ) -> IntegrityState:
        """
        Verifies the integrity of a file against a registered checksum using SHA-256.
        Always fails on a checksum mismatch.
        Fails on missing checksums based on the specified policy.
        """
        active_policy = policy or self.default_policy

        if not os.path.exists(filepath):
            raise ArtifactIntegrityError(f"Artifact file '{filepath}' does not exist.")

        # Compute current actual SHA-256 hash
        try:
            actual_sha = calculate_sha256(filepath)
        except Exception as e:
            raise ArtifactIntegrityError(f"Failed to calculate artifact checksum: {str(e)}") from e

        # If checksum is registered, it must match exactly
        if registered_sha256:
            if actual_sha == registered_sha256:
                return IntegrityState.VERIFIED
            else:
                raise ArtifactIntegrityError(
                    f"SHA-256 checksum mismatch for '{filepath}'. "
                    f"Expected: {registered_sha256}, Got: {actual_sha}"
                )

        # Handle missing checksums (legacy models)
        if active_policy == IntegrityPolicy.STRICT:
            raise ArtifactIntegrityError(
                f"Missing checksum for '{filepath}' rejected under STRICT policy."
            )
        elif active_policy == IntegrityPolicy.PRODUCTION_STRICT:
            if model_status == ModelStatus.PRODUCTION:
                raise ArtifactIntegrityError(
                    f"Missing checksum for PRODUCTION model '{filepath}' rejected under PRODUCTION_STRICT policy."
                )
            return IntegrityState.UNVERIFIED_LEGACY
        else:  # LENIENT
            return IntegrityState.UNVERIFIED_LEGACY
