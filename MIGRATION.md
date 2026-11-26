# MIGRATION — de-duplicating the deterministic pre-LLM layers

Three layers independently grew the same primitives: **dsc** (spine,
243 tests), **gama-fastpath** (HTTP API + gate, 40 tests) and **gama-offline**
(cache guards / CBR, 32 tests). This note records what has been merged, what is
still duplicated, and the order the rest must follow.

Acceptance rule for the merge:

```bash
# one copy of each primitive, and it lives in dsc
grep -rn "def parse_amount" dsc                 # amount.py
grep -rln "^T1 = \[" dsc                        # gate.py
grep -rn "def request_key" dsc                  # cachekey.py

# no caller keeps a private copy of any of them
grep -rn "def parse_amount\|def request_key\|^T1 = \[" gama-fastpath gama-offline
grep -rn "_FRESH = \|_MUTATING = \|_MULTI = " gama-fastpath gama-offline

grep -rn "hashlib.md5" gama-offline/gama_offline/  # only pretriage's process-local dedup
```

## Done

| primitive | before | after | why this copy won |
| --- | --- | --- | --- |
| money-field parsing | `gama-fastpath/fastpath.py:parse_amount` | `dsc/amount.py:parse_amount` | handles BHS/EU/US formats, currency tokens, negatives; pinned by `test_parse_amount`, `test_pdv_formats`, `test_pct_formats` |
| gate decision | `gama-fastpath/gate.py:check` + `T1`/`T2` in `fastpath.py` | `dsc/gate.py:check_structural`, `check_policy`, `check` | 18 reproduced bypasses plus 12 benign controls; parses argv instead of matching phrases |
| policy phrase lists | `gama-fastpath/fastpath.py:T1,T2` | `dsc/gate.py:T1,T2` (kept as `T1C`/`T2C`) | byte-identical to the fastpath copy, including `potvrdi da si root` and `podijeli\|podeli` |
| offline risk lists | `gama_offline/risk.py:T1,T2,T1C,T2C` | `dsc/gate.py:T1,T2,T1C,T2C`, re-exported by `risk.py` | one list; the diff is exactly the two documented widenings, frozen in `tests/test_risk_policy.py` |
| **cache key** | `fastpath.cache_key` (sha256 of `{"v","t","s"}`), `dsc.cache.SemanticCache._exact_key` (sha256 of `sid\|qc\|env:string`), `gama_offline.cache._key` (**md5** of `sid\|norm(q)`, no tenant, no context) | **`dsc/cachekey.py:request_key`** | `v2:<sha256>` over one canonical JSON object: `{v,t,tenant,session,ctx}`; all three call it; contract in `tests/test_cachekey.py` |
| **cacheability / freshness** | `gama_offline/cache.py:_FRESH` (regex), `_MULTI`, `_MUTATING` and the `if` chain in `cacheable()`; fastpath had no copy and could serve a stored clock answer | **`dsc/cachepolicy.py`** — `FRESH_RULES`, `MULTI_STEP_RULES`, `MUTATING_TOOLS`, `ERROR_MARKERS`, `block_reason` | one rule table, each row a named pattern with its effect and cost; `test_cachepolicy.py` walks the table, `gama-offline` `cache.fresh`/`cacheable` and `fastpath`'s cache read/store call it |
| **CBR intent key** | `gama_offline/cbr.py:_intent` — first six sorted unique words | **`request_key(goal, normalized=" ".join(content_tokens(goal)), context={"kind": "cbr"})`** | word order is significant, so unrelated goals no longer share a case; real corpus collision reproduced in `test_offline.py` |

`gama-fastpath` now depends on `dsc` (pinned commit, see its `requirements.txt`)
and keeps only what is specific to it: the HTTP surface, `pdv_na_neto`, `pct`,
`normalize`/`fold` and the SQLite store. `cache_key` is a one-line delegate, and
`cache_get` refuses any key that is not current-scheme.

`gama-offline` keeps `sid` scoping, `cacheable()` and `fresh()` as public names;
both are now delegates, and the rules live in `dsc.cachepolicy`. Only `_key`
changed for the cache key. Its `requirements.txt` pins the same dsc commit.

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

### The cacheability contract (`dsc.cachepolicy`)

`block_reason(res, text) -> str | None` returns a short reason or `None`:
`empty` (missing/short content, `MIN_ANSWER_LEN = 15`), `stopped`, `error`
(`(greska` / `error:` / `prekinuto nakon`), `fresh`, `mutating`, `multi_step`.
Order is frozen: shape → size → stopped → error → freshness → side effects →
multi-step. `text_block_reason(text)` is the text-only subset for a cache read
or a store that has no response yet.

A rule is a row, not an `if`:

| rule | effect | fires on |
| --- | --- | --- |
| `NADZOR_FRESH_V1` | fresh | live search/price/state words (`pretrazi`, `cijena`, `kurs`, `danas`, `screenshot`, …) |
| `ERP_FRESH_V2` | fresh | `zalih`, `lager`, `na stanju`, `saldo`, `dugovanj`, `kolicin`, `zadnja cijena`, … |
| `CLOCK_FRESH_V1` | fresh | `koliko je sati`, `koje je vrijeme`, `koji je datum`, `what time is it` |
| `CACHE_MULTI_GUARD` | multi_step | connector (`i`, `pa`, `zatim`, `;`, `&`) + action verb within 80 chars |
| `STATE_CHANGING_TOOL` | mutating | `tool_log` contains `mail`, `write`, `bash`, `postis`, `kasa`, … |

**Widening, on purpose:** `CLOCK_FRESH_V1` is new. The old `_FRESH` regex did not
catch a clock question, and fastpath checked its cache *before* its own time
route, so a stored `koje je vrijeme` could be replayed later. `koliko je sati`
is now a `fresh` miss in both layers. This is the only behavioural widening;
every phrase the old `_FRESH` matched still matches, byte-for-byte.

fastpath now consults the policy twice: `/optimize` skips the cache read for a
fresh/multi-step text (a row written before the policy existed stays in the
table and is ignored), and `/store` refuses to write one, returning
`{"ok": true, "stored": false, "reason": "fresh"}` instead of the old bare
`{"ok": true}`. Math routes are computed before the cache, so they are
unaffected.

### The CBR key (`gama_offline.cbr._intent`)

`intent` is now `request_key(goal, normalized=" ".join(content_tokens(goal)),
context={"kind": "cbr"})` — `v2:<sha256>` like every other key:

* same goal (case/whitespace/diacritics aside) → same case id;
* different goal → different id, because token **order** is part of the key;
* the old key was the first six *sorted unique* words, so two goals that merely
  used the same vocabulary collided. Found in the real `~/gamacode/cases.db`
  (ids 659, 660, 990): three unrelated multi-step goals — a `free_api`/QR/Word/
  PDF chain, a fleet-check chain, and a CSV sum — all keyed as `1 2 3 4 5 6`
  and would have been promoted as one recipe. `test_offline.py` reproduces that
  collision with the old algorithm and asserts three distinct keys now.
* no migration of `cases.intent`: old rows keep their plain-text intent, new
  rows get `v2:`. `promote()` therefore reports old buckets and new buckets
  separately until old rows age out. `find()` is unchanged (Jaccard on `_norm`)
  so recall does not move.

## Remaining duplicates (not touched yet)

1. **`gama_offline/pretriage.py` process-local dedup.** `_dcache` keys decisions
   with `md5(text)` in a plain dict (not persisted, not the answer cache). It
   is a different primitive and stays out of this migration for now; a future
   pass can use `request_key(..., context={"app": "pretriage"})` if it ever
   needs to survive a restart.

## Merge readiness

Everything the three packages duplicated is now in one place:

| primitive | owner | callers |
| --- | --- | --- |
| money parsing | `dsc/amount.py` | fastpath |
| gate + T1/T2 | `dsc/gate.py` | fastpath, offline (`risk.py` re-export) |
| cache key | `dsc/cachekey.py` | dsc cache, fastpath, offline, CBR intent |
| cacheability | `dsc/cachepolicy.py` | fastpath, offline |
| text normalisation | `dsc/normalize.py` | dsc; offline uses `content_tokens` for the CBR key and keeps `_norm` for recall scoring |

What still argues against a single package today (all deliberate, none
blocking):

1. `pretriage`'s `md5` dedup is process-local and unrelated; it can stay in
   offline even after a merge.
2. offline and fastpath each pin dsc by commit and keep their own CI; merging
   would need one release cadence and one pin.
3. `gama-offline` keeps `_norm` for `find()`, and fastpath keeps `norm`/`fold`
   for display. Both are documented divergences, not duplicates of the key.
4. packaging: dsc is `dsc`, the others are an app dir (`fastpath.py`) and a
   package (`gama_offline`); a merge means choosing entry points, not moving
   code.

With those four accepted, the code has no second copy left; the merge is a
packaging decision, not a correctness one.

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
- `dsc.normalize.is_fresh`/`is_multi_step` are *not* the policy. They match a
  token set and a diacritic-sensitive regex for `features()`/`SemanticCache`;
  `cachepolicy` is the replay rule (phrase regexes, diacritic-tolerant,
  mutating tools, response guards). Merging the two would silently narrow or
  widen one of the caches, which is why they stay separate and tested apart.
