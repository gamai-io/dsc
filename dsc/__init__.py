"""DSC Deterministic Semantic Core.

Offline, stdlib-first: normalize -> fingerprints -> deterministic vector -> lexical
hard checks -> calibrated verification -> two-stage semantic cache -> evidence detector.
"""
from .accounting import (
    fifo, fx, pdv_exclusive, pdv_inclusive, verify_arithmetic, verify_fx_claim, verify_pdv_claim,
)
from .api import Core
from .gtl import ToolSpec, TrustLayer
from .model import DeterministicModel
from .unify import Unified
from .cache import SemanticCache
from .detector import (
    fleet_determinism, load_key, poa_chain, poa_record, semantic_drift, verdict,
    verify_anchor, write_anchor,
)
from .fingerprints import hamming, jaccard_est, minhash128, sha256_hex, simhash64
from .fusion import calibrate_threshold, conformal_quantile, decide, ece, wrrf
from .hdc import cosine_q, dense, vector
from .lexical import BM25Index, hard_check, lexical_entailment
from .nondeterminism import (
    LEVELS, Envelope, compare as compare_receipts, describe as determinism_describe, guard,
    local_vllm_env, receipt, remote_api_env, required_ok, verify_receipt,
)
from .normalize import canonical, features, fold, parse_number
from .sigil import sigil_receipt, to_sigil, verify_sigil
from .verify import ConformalGate, split_claims, verify_answer, verify_claim

__all__ = [
    "Core", "SemanticCache", "poa_record", "poa_chain", "fleet_determinism",
    "semantic_drift", "verdict", "write_anchor", "verify_anchor", "load_key", "hamming", "jaccard_est", "minhash128",
    "sha256_hex", "simhash64", "calibrate_threshold", "conformal_quantile",
    "decide", "ece", "wrrf", "cosine_q", "dense", "vector", "BM25Index",
    "hard_check", "lexical_entailment", "canonical", "features", "fold",
    "parse_number", "ConformalGate", "split_claims", "verify_answer", "verify_claim",
    "Envelope", "receipt", "verify_receipt", "compare_receipts", "guard", "required_ok",
    "local_vllm_env", "remote_api_env", "determinism_describe", "LEVELS",
    "sigil_receipt", "to_sigil", "verify_sigil",
    "pdv_exclusive", "pdv_inclusive", "fifo", "fx", "verify_arithmetic",
    "verify_pdv_claim", "verify_fx_claim",
    "TrustLayer", "ToolSpec", "Unified", "DeterministicModel",
]
