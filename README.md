# dsc — Deterministic Semantic Core + GTL/1

Offline, stdlib-first layer for agent decisions: normalize → deterministic fingerprints → lexical retrieval → hard checks → calibrated cache → proof of action.

## Why

Two failure modes dominate agent systems: **false semantic hits** (high cosine similarity that still passes contradictions) and **stale facts** (cached prices/stock served after they changed). `dsc` closes both with rules, not with a bigger model: canonical fingerprints, exact-number verification, and a two-stage cache with provenance.

## Components

- `dsc/normalize.py` — NFC + diacritics + numbers/units + negation, deterministic
- `dsc/fingerprints.py` — simhash64 / minhash128 / sha256 over canonical text
- `dsc/hdc.py` — int8 hyperdimensional document vectors (offline)
- `dsc/model.py` — BM25 stats + HDC vectors + per-category centroids
- `dsc/lexical.py`, `dsc/fusion.py` — lexical ranking + RRF fusion
- `dsc/cache.py` — two-stage semantic cache (exact → similarity) with TTL and guards
- `dsc/verify.py`, `dsc/detector.py` — claim verification, PoA/poisoning detectors
- `dsc/gtl.py` — GTL/1 trust layer: tool envelopes, receipts, replay, audit
- `dsc/accounting.py` — Decimal money math (VAT, FX, margins), round-half-up

## Install / run

```bash
# from repo root
python3 -m unittest discover -s tests -t . -v
python3 -m dsc.cli analyze "VAT is 17% in BiH"
python3 -m dsc.bench
```

No network, no GPU, no randomness: same text → same bytes on every node.

## Reproduce

`python3 -m unittest discover -s tests -t .` — 193 tests, no external services.

## Limitations

- Fingerprints are lexical; they do not replace embeddings, they gate them.
- The cache is per-process SQLite; distributed deployments need a shared store.
- Money math covers VAT/FX/margins only; statutory rules are your responsibility.

## License

Apache-2.0.
