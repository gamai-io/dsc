# dsc — DSC Deterministic Semantic Core + GTL/1 (v0.3)

> **Nije Microsoft Semantic Kernel (SK).** SK je orchestrator za AI aplikacije (C#/Python/Java).
> DSC je deterministički sloj odluke: normalizacija + otisci + hard-check + keš + dokaz.
> Paket se zove `dsc` upravo da se ne miješa sa „semantic kernel“ proizvodom.

Offline, stdlib-first sloj koji spaja: normalizaciju, determinističke otiske, vektore,
leksički retrieval, hard-check verifikaciju, dvoslojni semantički keš i dokaz (PoA/FDS/SDM).
Bez mreže, bez GPU, bez `hash()` randomizacije. Isti tekst → isti bajtovi na svakom čvoru.

## Zašto

Postojeći DSC sloj ima 4 odvojena keša i 5 embedding klijenata. Dva najveća rizika:
lažni semantički HIT (kosinus 0.93 propušta i kontradikcije, izmjereno 0.929–0.949) i
stale zalihe/cijene (globalni TTL 900–3600 s). Ovaj paket ih zatvara pravilima, ne modelom.

## Instalacija / pokretanje

```bash
# from repo root
python3 -m unittest discover -s dsc/tests -t . -v
python3 -m dsc.bench
python3 -m dsc.cli analyze "PDV je 17% u BiH"
python3 -m dsc.cli sim "koliko ima kupaca" "koliko kupaca ima"
python3 -m dsc.cli verify --answer "Graco 795 ima 3300 PSI [1]." \
  --source "Graco Ultra 795: 3300 PSI, 0.95 GPM."
python3 -m dsc.determinism_check --url http://127.0.0.1:8100/v1 \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ --batch-sizes 1,8,32
python3 -m dsc.lab_eval --data /tmp/lab_embed/data85.json \
  --embed-url http://127.0.0.1:8064/v1/embeddings --bootstrap
python3 -m dsc.golden_expand --out docs/dsc/golden_synth300.json
python3 -m dsc.lab_eval --data docs/dsc/golden_synth300.json \
  --embed-url http://127.0.0.1:8064/v1/embeddings --bootstrap
python3 -m dsc.cache_calib --data docs/dsc/golden_synth300.json
# human graded labels (local web page)
python3 -m dsc.label_tool build --data docs/dsc/golden_synth300.json --max-queries 100
python3 -m dsc.label_tool serve --port 8977
python3 -m dsc.label_tool export --out docs/dsc/golden_graded.json
```

`determinism_check` vraća `cross_batch_equal=true/false`; exit code 2 znači da endpoint
**nije** batch-invariant (isti prompt daje različit izlaz pod različitim opterećenjem).

Python API:

```python
from dsc import Core
core = Core(cache_path="/tmp/dsc.db")          # embed_fn=... za bge-m3 (opciono)
core.remember("koliko ima kupaca u bazi", "1114 kupaca", sid="s1", domain="doc")
core.recall("koliko kupaca ima u bazi", sid="s1", domain="doc")   # semantic HIT
core.verify("Graco 795 ima 3300 PSI [1].", [{"id": 1, "text": "Graco 795: 3300 PSI"}])
```

## Invarijante

1. **Nema lažnog SUPPORTED.** Brojevi/jedinice/datumi/polarnost se provjeravaju prije NLI-a;
   kontradikcija je veto, ne glas.
2. **Keš nije dokaz.** Svaki HIT nosi `cached_at`, `expires_at`, `epoch`, `sources`,
   `evidence_id`, `trust`. Bez `trust="verified"` nema semantičkog HIT-a.
3. **Brojevi su exact-only.** Novac/procenat/fizičke jedinice/decimalni brojevi nikad ne idu
   kroz semantički HIT; samo kroz kanonski tačan ključ + kratki TTL.
4. **Determinizam.** BLAKE2b/SHA-256, bez `hash()`, bez locale, bez randoma; vektor je int8.
5. **Fallback.** Ako NLI/embedding/Qdrant padnu, sistem daje konzervativniju odluku
   (`UNVERIFIED`), nikad lažni `SUPPORTED`.

## Matematika (kratko)

- **Kanonizacija:** NFC → mojibake fix → fold dijakritike → brojevi u `#<decimal>` →
  jedinice u dimenziju (`pressure/power/volume/mass/length/time/currency/percent`).
- **SimHash:** `E[H] = b·θ/π`; prag za cos 0.90 pri `b=64` je `τ_H ≈ 13` (ne 3).
- **MinHash:** `P[min jednak] = J`; 128 permutacija, greška `sqrt(J(1−J)/128)`.
- **HDC vektor:** hashed 3–5 gram + riječi → signed random projection → L2 → int8.
  Kosinus se računa na int8 (skala se skraćuje).
- **BM25-lite:** `k1=1.5`, `b=0.75`, IDF `log(1+(N−df+0.5)/(df+0.5))`.
- **Verifikacija:** `z = β + w_lex·lex + w_emb·cos + w_nli·entail − w_con·contra`,
  `P = σ(z)`. Uz `β=−2.8, w=(3.0, 2.0, 2.8)`. Hard-check i kontradikcija imaju veto.
  Conformal: prag = kvantil `⌈(n+1)(1−α)⌉/n` negativnih skorova.
- **Fusion:** `wRRF(d) = Σ w_i/(k + rank_i(d))`, `k=10` (ne 60), težine kalibrisane.
- **Keš:** kanonski SHA-256 (exact) → SimHash kandidati (Hamming ≤ 18) → int8 kosinus
  (prag 0.72 bez modela / 0.86 sa bge-m3) → leksički gate (Jaccard ≥ 0.34) → brojevi/polarnost.
- **Detektor:** PoA lanac (SHA-256 payload + `prev`), FDS (pairwise Jaccard tragova +
  permutacijski test), SDM (`1 − mean(conf·(1−contra)·2^(−age/half_life))`).

## Mjereno (Contabo, CPU, 2026-10-25)

| Operacija | p50 | p95 |
|---|---|---|
| normalize | 0.25 ms | 0.32 ms |
| canonical | 0.02 ms | — |
| SimHash | 1.05 ms | — |
| HDC vektor | 0.57 ms | — |
| cache put | 1.33 ms | 1.73 ms |
| cache exact | 0.05 ms | — |
| cache semantic | 1.26 ms | — |
| cache miss | 0.11 ms | — |
| verify | 1.33 ms | 1.81 ms |
| inference receipt | 0.05 ms | — |
| receipt verify | 0.08 ms | — |
| lab_eval (85 upita, bez bge) | ~0.2 s | — |
| cache_calib (108 probes) | ~2 s | — |
| label_tool build (100 queries) | ~1 s | — |

Kvalitet na sintetičkom setu: exact/semantic HIT 100%, **0 lažnih HIT-ova** na 4 tvrda
negativa (broj/zaliha/negacija/jedinica), verifikacija 100% na supported / wrong-number /
negation. Determinizam potvrđen kroz dva odvojena procesa.

## Lab golden set (85 upita / 40 dok, bosanski)

`lab_eval` reprodukuje lab BM25 **1:1** i mjeri DSC offline:

| Metoda | R@1 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| lab BM25 (baseline) | 0.8471 | 0.8990 | 0.9138 |
| BM25 (DSC) | 0.8941 | 0.9416 | 0.9553 |
| HDC int8 | 0.8824 | 0.9283 | 0.9426 |
| **BM25 + HDC** | **0.9176** | **0.9539** | **0.9657** |
| bge-m3 (Contabo) | 0.9529 | 0.9745 | 0.9811 |
| BM25+HDC+bge | **0.9647** | **0.9804** | **0.9835** |

Paired bootstrap: HDC+BM25 **značajno** diže nDCG nad BM25 (p=0.001 on the 324 stress set;
p=0.0105 on the 85 real set); above bge-m3 the 85-set gain is not significant (p≈0.84),
while the synthetic 324-set triple fusion is significant (p=0.047 R@1 / 0.026 nDCG).
Treat the synthetic set as regression, not proof.

Cache threshold calibration (108 variants / 40 negatives): 0.72 → 97.8% recall, 2.5% FPR;
**0.80 → 96.7% recall, 0% FPR**; 0.86 → 87.9%. `SemanticCache.PROFILES` exposes
`balanced / strict / paranoid`. Details: `docs/dsc/D-lab-eval.md`.

## GTL/1 — one trust layer for every tool

```python
from dsc.unify import Unified
u = Unified(cache_path="/tmp/c.db", gtl_path="/tmp/gtl.db",
            model_path="docs/dsc/dsc_model.json")
r = u.call("calc", {"expr": "(2+3)*4"})          # receipt + trust=EXACT
u.call("remember", {"query": "...", "answer": "...", "sid": "s"},
       idem_key="unique-key")                     # side effect: atomic idempotency
u.verify(r["receipt"], result=r["result"])        # hash/chain/signature check
```

- Every call → PoA receipt; optional HMAC chain and Ed25519 signature.
- Side effects require `idem_key`: atomic claim, replay returns the recorded result, never re-executes.
- Cache only for deterministic non-side-effect tools; errors are never cached; cache key includes `tool_version`.
- Unbound `EXACT` inference is clamped to `VERSIONED` (`inference_bound`).
- Chain state persists across restart (`chain_state`).
- Signed inference: `sign_inference(receipt(...), private_key)` + `TrustLayer(inference_public_key=pub)`;
  an unverified `EXACT` is clamped to `VERSIONED`.
- MCP adapter: `python3 -m dsc.mcp_server --model docs/dsc/dsc_model.json`
  (`initialize`, `tools/list`, `tools/call` → `structuredContent` carries the receipt).
- Signed inference proxy: `python3 -m dsc.inference_service --upstream http://127.0.0.1:8100
  --model Qwen/Qwen2.5-7B-Instruct-AWQ --sign-key /secure/infer.pem --port 8094`; `POST /infer`
  returns `{output,envelope,receipt,trust}` with an Ed25519 signature.
- Spec: `docs/dsc/GTL_SPEC.md`; critics: `docs/dsc/E-critics.md`.

## Deterministic model (trained, offline)

```bash
python3 -m dsc.train --data /tmp/lab_embed/data85.json --out docs/dsc/dsc_model.json --eval
```

`DSC-DSC-0.1`: BM25 + int8 HDC + category centroids, 77 KB JSON, R@1 0.9059 on the lab set.
No GPU, no LLM. It is a retriever/classifier, not a chat model.

## Nondeterminizam LLM-a (Thinking Machines, sep 2025)

Ključni nalaz: **`temperature=0` nije dovoljno**. Uzrok nije „konkurencija + floating point“,
nego **nedostatak batch invariance**: izlaz jednog zahtjeva zavisi od toga koji je batch
(broj i sastav paralelnih zahtjeva) u datom trenutku. Ista postavka na Qwen3-235B dala je
**80 različitih kompletacija od 1000** na temp 0; sa batch-invariant kernelima **1/1000**.
Cijena: ~26 s → 42 s za 1000 sekvenci (1.6×), uz optimizovan attention kernel.

DSC to ne može popraviti na tuđem API-ju, ali može **spriječiti lažni dokaz**:

- `Envelope` bilježi engine/verziju/model/hash/kvantizaciju/preciznost/uređaj/**batch_size**/
  **batch_signature**/seed/temperature.
- `level()`: `EXACT` (batch-invariant + deterministic kernels) · `BATCH_INVARIANT` ·
  `VERSIONED` (temp 0 + vezan batch) · `STOCHASTIC`.
- `receipt(prompt, output, env)` + `verify_receipt` vezuju izlaz za envelope; `replay_key`
  se mijenja kad se promijeni batch (osim ako je batch-invariant).
- Keš je vezan za `envelope_id`: odgovor iz batcha A **nikad** se ne servira za batch B.
  Stohastički odgovori su **exact-only** (nema semantičkog HIT-a).
- `verdict(..., inference=..., requirement="EXACT")` → `DOUBTFUL` ako je izvor samo
  `VERSIONED`/`STOCHASTIC`.

```python
from dsc import Core, local_vllm_env, receipt, guard
env = local_vllm_env("qwen7b", batch_size=8, batch_signature="a,b,c")
rec = receipt("pitanje", "odgovor", env)          # level=EXACT
core = Core(cache_path="/tmp/dsc.db")
core.remember("pitanje", "odgovor", inference=rec)  # vezano za envelope
core.recall("pitanje", inference=rec)               # HIT samo za isti envelope
guard(env, "EXACT")                                 # ok=True
```

Za remote API: `remote_api_env("deepseek", "model", batch_size=?, batch_signature=?)`
→ `VERSIONED`; bez `batch_signature` → `replay_bound=False`.

**Prvi realni probe (Vast vLLM, Qwen2.5-7B-AWQ, temp 0):** prvi prolaz 1/8/16 konkurentno
dao je 2 unikatna izlaza; kontrolni 120 uzoraka na concurrency 24 dao je **120/120 identično**
(0 grešaka). Zato: jedan probe nije dokaz; mjeri matricu i koristi batch-invariant kernele za `EXACT`.

## Dokaz: HMAC lanac + anchor (iz Protean laba)

`poa_record(..., key=..., prev_chain=..., seq=...)` + `poa_chain(..., key=..., anchor=...)`
+ `write_anchor/verify_anchor` + `load_key` (0600). Lanac je
`HMAC(key, prev_chain || sha256(canon(payload)))`; anchor je append-only fajl `seq chain`.
Brisanje zapisa, izmjena polja i skraćivanje anchor fajla se detektuju.

## Optional DSC profile: accounting + SIGIL

- `pdv_exclusive/pdv_inclusive/fifo/fx` — Decimal, ROUND_HALF_UP. **Opt-in only.**
- `verify_arithmetic` is generic and wired into `hard_check`; a wrong inline calculation
  vetoes the claim (`17% od 2400 = 480` fails, `= 408` passes). This is math, not DSC business.
- PDV/FIFO/FX checks run only with `accounting=True` (`Core.verify(..., accounting=True)`);
  the neutral core knows nothing about PDV.
- `sigil_receipt(poa, policy, intent, anchor)` maps PoA to SIGIL/1-style fields;
  `verify_sigil` checks `receipt_hash`, `result_digest`, `intent_hash`, `policy_hash`.
- **Ed25519 signatures** (optional): `generate_keypair`, `sign_receipt`, `verify_sigil(..., public_key=..., require_signature=True)`.
  Private key is written `0600`; tampering any receipt field fails `E_SIGNATURE` + the field digest.

## Human graded labels

`label_tool` builds a queue (BM25+HDC top-k per query, gold forced in), serves a local page
at `127.0.0.1:8099`, saves JSONL, and exports a `lab_eval`-compatible file with `graded` gains.
Pre-labels: gold=2, same category=1, else 0; the human corrects them.

## Ograničenja (ne tvrdimo)

- HDC **nije** zamjena za bge-m3: hvata leksiku, ne sinonimiju. Služi kao offline fallback
  i za keš kandidate.
- Bez kalibrisanog bosanskog seta (300–500 parova) pragovi su inženjerski, ne statistički
  garantovani. `ConformalGate` daje garanciju tek nakon `fit`.
- FDS mjeri reproducibilnost tragova, ne istinitost akcije.
- PoA dokazuje integritet zapisa i lanac, ne ispravnost izvršenja.
- `EXACT` za LLM je moguć samo sa batch-invariant kernelima (npr. vLLM + `batch_invariant_ops`);
  za cloud API realno je `VERSIONED` ili `STOCHASTIC`, i tako se i označava.

## Integracija (predlog, bez presijecanja produkcije)

```python
# u harnessu, iza flag-a DSC_DSC=1
from dsc import Core
_dsc = Core(cache_path="/opt/dsc-harness/_dsc.db")
hit = _dsc.recall(text, sid=sid, domain=_dom(text))
if hit:
    return hit  # nosi meta.evidence_id, cached_at, epoch, trust
...
_dsc.remember(text, answer, sid=sid, domain=_dom(text), sources=src, evidence_id=eid)
```

Redoslijed migracije: (1) dual-write stari + novi keš, (2) read kroz novi sa fallbackom,
(3) ugasi stari semcache, (4) uključi evidence gate, (5) kalibracija na golden setu.

## Polimorfija (jedan ugovor, više backendova)

Isti `Core` ugovor radi sa bilo kojim backendom; deterministički sloj je uvijek prisutan.

```python
from dsc.backends import auto
b = auto(embed_url="http://127.0.0.1:8064/v1/embeddings",
         nli_url="http://127.0.0.1:8090")   # probe; ako padne -> None
core = Core(embed_fn=b["embed_fn"], nli_fn=b["nli_fn"])
```

- `embed_fn=None` → HDC int8 (offline, deterministički).
- `embed_fn=bge-m3` → semantički retrieval/keš (mreža, opciono).
- `nli_fn=None` → hard-check + leksički/HDC signali; `UNVERIFIED` umjesto lažnog `SUPPORTED`.
- `nli_fn=NLI` → contradiction veto + entailment signal.
