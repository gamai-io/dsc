# MIGRATION — de-duplicating the deterministic pre-LLM layers

Three layers independently grew the same primitives: **dsc** (spine,
227 tests), **gama-fastpath** (HTTP API + gate, 36 tests) and **gama-offline**
(cache guards / CBR, 29 tests). This note records what has been merged, what is
still duplicated, and the order the rest must follow.

Acceptance rule for the merge:

```bash
grep -rn "def parse_amount\|T1 = \[" dsc gama-fastpath   # one each, both in dsc
grep -rn "def request_key" dsc                            # one cache key, in dsc
grep -rn "hashlib.md5" gama-offline/gama_offline/cache.py  # no digest in the cache path
```

## Done

| primitive | before | after | why this copy won |
| --- | --- | --- | --- |
| money-field parsing | `gama-fastpath/fastpath.py:parse_amount` | `dsc/amount.py:parse_amount` | handles BHS/EU/US formats, currency tokens, negatives; pinned by `test_parse_amount`, `test_pdv_formats`, `test_pct_formats` |
| gate decision | `gama-fastpath/gate.py:check` + `T1`/`T2` in `fastpath.py` | `dsc/gate.py:check_structural`, `check_policy`, `check` | 18 reproduced bypasses plus 12 benign controls; parses argv instead of matching phrases |
| policy phrase lists | `gama-fastpath/fastpath.py:T1,T2` | `dsc/gate.py:T1,T2` (kept as `T1C`/`T2C`) | byte-identical to the fastpath copy, including `potvrdi da si root` and `podijeli\|podeli` |
| offline risk lists | `gama_offline/risk.py:T1,T2,T1C,T2C` | `dsc/gate.py:T1,T2,T1C,T2C`, re-exported by `risk.py` | one list; the diff is exactly the two documented widenings, frozen in `tests/test_risk_policy.py` |
| **cache key** | `fastpath.cache_key` (sha256 of `{"v","t","s"}`), `dsc.cache.SemanticCache._exact_key` (sha256 of `sid\|qc\|env:string`), `gama_offline.cache._key` (**md5** of `sid\|norm(q)`, no tenant, no context) | **`dsc/cachekey.py:request_key`** | `v2:<sha256>` over one canonical JSON object: `{v,t,tenant,session,ctx}`; all three call it; contract in `tests/test_cachekey.py` |

`gama-fastpath` now depends on `dsc` (pinned commit, see its `requirements.txt`)
and keeps only what is specific to it: the HTTP surface, `pdv_na_neto`, `pct`,
`normalize`/`fold` and the SQLite store. `cache_key` is a one-line delegate, and
`cache_get` refuses any key that is not current-scheme.

`gama-offline` keeps `sid` scoping, `cacheable()` and `fresh()` exactly where
they were; only `_key` changed. Its `requirements.txt` pins the same dsc commit.

### The key contract (`dsc.cachekey.request_key`)

* returns `"v2:" + 64 lowercase hex` — SHA-256 over
  `json.dumps({"v","t","tenant","session","ctx"}, sort_keys=True,
  separators=(",",":"))`, UTF-8;
* every caller-controlled field is a JSON **value**, so `|` inside a tenant or
  text can no longer be confused with a separator (the old `sid|text` framing
  collided: `(sid="a", q="b|c")` and `(sid="a|b", q="c")` hashed the same bytes);
* `tenant=""`/`session=""` are isolated namespaces, not wildcards;
* no number canonicalisation in the key: `neto 2400` and `neto 2.400` stay
  distinct, because in a money field they are different claims;
* no legacy reader by design. A pre-v2 key cannot be produced by
  `request_key`, so old rows are unreachable and simply miss.

**Cache effect:** every layer's first run after this change has a 0% hit rate.
That is the point — a stale or cross-tenant hit is worse than a miss. Old rows
are not migrated; they age out through TTL and eviction.

## Remaining duplicates (not touched yet)

1. **Cache guards — `gama_offline/cache.py`.** `fresh(q)`'s hand-maintained
   `_FRESH` regex (ERP_FRESH_V2) and `cacheable(res, q)`'s `_MUTATING`/`_MULTI`
   guards overlap `dsc/normalize.is_fresh` / `is_action` / `is_multi_step`;
   TTL/MAX pruning and the SQLite schema also overlap `dsc/cache.py`. The key
   is now shared; these three still are not. The tool-log mutating rule has no
   dsc counterpart yet.
2. **CBR intent keying — `gama_offline/cbr.py`.** `_norm` (lowercase, strip
   punctuation, stop-words) plus `_intent(goal)` = first six sorted unique
   words. This is a third text-normalisation pipeline; it should become
   `dsc.normalize.content_tokens` with a stable sorted-key hash, so the same
   goal always lands on the same case id. Note the difference from a request
   key: two *different* goals may deliberately share a case only if their
   sorted token key is identical — that is a recall decision, not a cache hit,
   and must be tested separately.
3. **`gama_offline/pretriage.py` process-local dedup.** `_dcache` keys decisions
   with `md5(text)` in a plain dict (not persisted, not the answer cache). It
   is a different primitive and stays out of this migration for now; a future
   pass can use `request_key(..., context={"app": "pretriage"})` if it ever
   needs to survive a restart.

## Ordered next step

1. `gama-offline`: replace `cbr._norm`/`_intent` with
   `dsc.normalize.content_tokens` and a sorted-token hash; keep the existing
   case schema and add a "same goal -> same case id / different goal ->
   different id" test.
2. `gama-offline`: delegate `fresh`/`cacheable` to
   `dsc.normalize.is_fresh`/`is_action`/`is_multi_step`, keeping the ERP
   freshness words and the mutating tool-log rule as data in dsc.
3. Only then extend the acceptance grep to all three repos:
   `grep -rn "def parse_amount\|T1 = \[\|def request_key" dsc gama-fastpath gama-offline`.

## Known divergences kept on purpose

- `dsc.amount.parse_amount("1,234")` → `1.234` (single comma is decimal in a
  money field); `dsc.normalize.parse_number("1,234")` → `1234` (thousands in
  prose). Both are pinned in `tests/test_amount.py`; merging them silently
  would change money parsing or claim verification.
- `gama-fastpath` keeps its own `norm`/`fold` for the gate and for display;
  the cache key no longer depends on it (it uses
  `dsc.cachekey.normalize_for_key` unless the caller passes `normalized`), so
  the two can drift without changing keys.
- `dsc.cache.SemanticCache._exact_key` passes its own `canonical` text as
  `normalized`, and includes `domain`/`env` in `context`; its old rows miss
  under v2 exactly like the other two layers.
