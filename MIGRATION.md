# MIGRATION — de-duplicating the deterministic pre-LLM layers

Three layers independently grew the same three primitives: **dsc** (spine,
211 tests), **gama-fastpath** (HTTP API + gate, 27 tests) and **gama-offline**
(cache guards / CBR). This note records what has been merged, what is still
duplicated, and the order the rest must follow.

Acceptance rule for the merge:

```bash
grep -rn "def parse_amount\|T1 = \[" dsc gama-fastpath   # one each, both in dsc
```

## Done

| primitive | before | after | why this copy won |
| --- | --- | --- | --- |
| money-field parsing | `gama-fastpath/fastpath.py:parse_amount` | `dsc/amount.py:parse_amount` | handles BHS/EU/US formats, currency tokens, negatives; pinned by `test_parse_amount`, `test_pdv_formats`, `test_pct_formats` |
| gate decision | `gama-fastpath/gate.py:check` + `T1`/`T2` in `fastpath.py` | `dsc/gate.py:check_structural`, `check_policy`, `check` | 18 reproduced bypasses plus 12 benign controls; parses argv instead of matching phrases |
| policy phrase lists | `gama-fastpath/fastpath.py:T1,T2` | `dsc/gate.py:T1,T2` (kept as `T1C`/`T2C`) | byte-identical to the fastpath copy, including `potvrdi da si root` and `podijeli\|podeli` |

`gama-fastpath` now depends on `dsc` (pinned commit, see its `requirements.txt`)
and keeps only what is specific to it: the HTTP surface, `pdv_na_neto`, `pct`,
`normalize`/`fold` and the SQLite cache with its SHA-256 key.

## Remaining duplicates (not touched yet)

1. **Cache guards — `gama_offline/cache.py`.** Four independent things that
   `dsc/cache.py` already does better or should own:
   - `_key(sid, q)` = `md5(sid|norm(q))` — md5, and a plaintext-free key is
     already solved in `dsc/cache.py:_exact_key` (`sha256_hex`, sid scope,
     `global_scope`, `envelope_id`). fastpath's `cache_key` is the better
     reference (SHA-256 of a canonical JSON payload, tenant in the key).
   - `fresh(q)` — a hand-maintained freshness regex (`_FRESH`, ERP_FRESH_V2
     additions) that overlaps `dsc/normalize.py:is_fresh` + `FRESH_WORDS`.
   - `cacheable(res, q)` — the skip-mutating / skip-multi-step guards
     (`_MUTATING`, `_MULTI`) that overlap `normalize.is_action` /
     `normalize.is_multi_step`; the tool-log rule has no dsc counterpart yet.
   - TTL/MAX pruning + SQLite schema — `dsc/cache.py` already has TTL, epochs
     and provenance, so the offline layer should delegate, not re-implement.
2. **CBR intent keying — `gama_offline/cbr.py`.** `_norm` (lowercase, strip
   punctuation, stop-words) plus `_intent(goal)` = first six sorted unique
   words. This is a third text-normalisation pipeline; it should become
   `dsc.normalize.content_tokens` with a stable sorted-key hash, so the same
   goal always lands on the same case id.
3. **`gama_offline/risk.py` still carries its own `T1`/`T2`.** They are the
   pre-2026-11-26 copy and are **not** identical to `dsc/gate.py`: risk.py's T2
   lacks the `podijeli|podeli` alternates in the env-disclosure pattern and
   lacks `potvrdi da si root`. Until it imports `dsc.gate.T1/T2` the acceptance
   grep over all three repos still shows two `T1 = [` lines (one in dsc, one
   here). This is deliberately out of scope for this commit.

## Ordered next step

1. `gama-offline`: delete `risk.py`'s `T1`/`T2`/`T1C`/`T2C` and
   `from dsc.gate import T1, T2, T1C, T2C`. Adopting dsc's list widens offline
   T2 by exactly two phrases (`podijeli|podeli` env disclosure, `potvrdi da si
   root`); add a gama-offline test for both, then pin.
2. `gama-offline`: extract `dsc/cachekey.py` (`key(sid, text)`,
   `fresh(text)`, `cacheable(result, text)`), move `_MUTATING` tool-log logic
   there with tests, and make both `gama_offline/cache.py` and
   `gama-fastpath/cache_key` call it.
3. `gama-offline`: replace `cbr._norm`/`_intent` with
   `dsc.normalize.content_tokens` and a sorted-token hash; keep the existing
   case schema.
4. Only then extend the acceptance grep to all three repos:
   `grep -rn "def parse_amount\|T1 = \[\|def cache_key" dsc gama-fastpath gama-offline`.

## Known divergences kept on purpose

- `dsc.amount.parse_amount("1,234")` → `1.234` (single comma is decimal in a
  money field); `dsc.normalize.parse_number("1,234")` → `1234` (thousands in
  prose). Both are pinned in `tests/test_amount.py`; merging them silently
  would change money parsing or claim verification.
- `gama-fastpath` keeps its own `norm`/`fold` because it is a two-line fold
  used before the gate and the cache key; moving it would drag
  `dsc.normalize.clean` (mojibake + NFC + ligatures) into the request path and
  change the cache key for already-stored entries.
