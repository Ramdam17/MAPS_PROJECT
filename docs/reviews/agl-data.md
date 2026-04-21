# Component review — `src/maps/experiments/agl/data.py`

**Sprint-08 D.27. Reviewer :** Rémy Ramadour + Claude, 2026-04-20.
**File under review :** `src/maps/experiments/agl/data.py` (334 L ; 1 enum + 1 dataclass + 6 functions).
**Paper sources :** paper §A.2 AGL, citing Dienes (1997) for grammar FSM.
**Student source :** `external/paper_reference/agl_tmlr.py:268-441` (grammar gen + encode + target_second).
**Callers :** `trainer.py:pre_train` L248-256 + L288 + `evaluate` L370-405.
**DoD :** read-only audit written. No code touched.

---

## (a) Architecture overview

| Port entity | Lines | Student counterpart |
|:--|:--|:--|
| `GrammarType` (IntEnum 1/2/3) | L62-71 | raw int `grammar_type` arg in `Array_Words` |
| `TrainingBatch` (dataclass, `patterns` only) | L74-84 | returned as bare tensor from `Array_Words` |
| `generate_random_word()` | L87-94 | `Generate_Word_Random` L268-275 |
| `generate_grammar_a()` | L97-165 | `Generate_Grammar_A` L278-311 (walrus ladder) |
| `generate_grammar_b()` | L168-231 | `Generate_Grammar_B` L315-351 |
| `encode_word(word)` | L234-248 | `encode_word` L361-383 |
| `generate_batch(grammar_type, number, device)` | L251-289 | `Array_Words(grammar_type, number, output)` L392-413 |
| `target_second(input, output)` | L292-334 | `target_second` L421-441 |

### Architectural gain over student
- `GrammarType` enum replaces magic-number `grammar_type=1/2/3`.
- `TrainingBatch` dataclass (single field `patterns`) — minimal but symmetric with Blindsight's 3-field TrainingBatch. Future-proofing.
- Walrus-ladder expanded into readable `if/elif` blocks — **bit-parity RNG-wise** preserved (same number of `random.randint` calls per iteration regardless of which branch fires).

---

## (b) Math / RNG parity — 8-step comparison

| Step | Student | Port | Match |
|:--|:--|:--|:-:|
| Word length | `random.randint(3, 8)` | `random.randint(3, 8)` | ✅ |
| Random-word letters | `random.choice(["x","v","m","t","r"])` × N | `random.choice(_ALLOWED_LETTERS)` × N (generator expr) | ✅ — same N calls, same order |
| Grammar A/B FSM | walrus ladder with 5 × `random.randint(1,2)` per iter | if/elif with 5 × `random.randint(1,2)` per iter | ✅ — same RNG consumption |
| Grammar A/B termination | `if position==6: break` (A) / `if position==5: ...` (B) | identical | ✅ |
| Letter encoding | `mapping.get(letter, [0]*6)` | `_LETTER_TO_ONEHOT.get(letter, (0,)*6)` | ✅ |
| Out-of-range chunk | `if end_index > 48: break` | `if end > NUM_INPUT_UNITS: break` | ✅ |
| Batch assembly | `while len(list_words) < number` | `for _ in range(number)` | ✅ — same N iterations |
| `target_second` top-k eq | `torch.topk(output[i], k); set(...)==set(...)` | identical + explicit k=0 guard | ✅ + port safer |

✅ **8 steps bit-parity RNG + math**. Aucune divergence numérique.

### ⚠️ Port improvement (not divergence)
Port `target_second` L326-330 guards `k == 0` explicitly (returns 1.0 for degenerate all-zero rows). Student would hit `torch.topk(x, 0)` which returns empty tensor — `set() == set()` → True → result = 1.0 anyway. **Same behaviour**, port just avoids the spurious empty-tensor allocation.

---

## (c) Alphabet + encoding constants

- `BITS_PER_LETTER = 6` ✅ student `bits_per_letter = 6` L1676.
- `MAX_WORD_LENGTH = 8` ✅ student word length ∈ [3, 8].
- `NUM_INPUT_UNITS = 48` ✅ student `encoded = [0] * 48` L370.
- **Legacy 6th bit** — always 0, documented in port L49-51 (comment: "kept for legacy compatibility with a version that once had a 6-letter alphabet"). Student silent on this.
- Letter ordering `{x, v, m, t, r}` ✅ student L363-367.

---

## (d) Subtle behaviour preserved

### (d1) Grammar A node 5 stalling
Student L309 `if position==6: break`. For Grammar A, positions 1-5 are live; position 6 is terminal.
Port L163 same. If the RNG happens to emit a Grammar A word with only node-1→2→4→4→4 transitions
and never hits node 6, the loop exits on `len(word) == number_letters`. Both codes behave
identically.

### (d2) Grammar B "stuck at node 5"
Student L345-349 : at position 5, path=1 appends "r" **without changing position**; path=2 + `len>2`
breaks; else the outer `while` continues (position stays 5 forever until length is met). Port
L226-230 identical. Quirky but preserved.

### (d3) Unknown letters encode as zero-chunk
Student L381 `mapping.get(letter, [0]*6)` — any letter not in {x,v,m,t,r} silently encodes to zeros.
Port L247 same. Not triggered in practice (only {x,v,m,t,r} ever produced by generators) but
defensive.

### (d4) Batch device placement
Student L412 `torch.Tensor(list_words).to(device)` uses module-global `device`.
Port L288 `torch.tensor(..., device=device)` accepts `device` as parameter → cleaner, test-friendly.
Numerically identical (`torch.Tensor` default dtype is float32, matches `torch.tensor(..., dtype=torch.float32)`).

---

## (e) Cross-references — related student logic NOT in data.py

| Student function | Role | Port location |
|:--|:--|:--|
| `calculate_metrics` L451-489 | TP/FP/TN/FN + return **only precision** (accuracy/recall/f1 computed but discarded) | inlined in `trainer.py:evaluate` L384-393 — precision-only, same as student |
| `compute_metrics` L1142-1148 | TP/TN-based accuracy for wagering | inlined in `trainer.py:evaluate` L406-410 (wager_accuracy) |

Both are fine; the port inlines these trivial counters directly in `evaluate()`. Worth noting
that `calculate_metrics` computes 4 metrics but returns 1 — a minor student code smell. If we
ever want recall/F1 reported, we'd expand `evaluate()` (deferrable — not a reproduction blocker).

---

## (f) RG-003 relevance

**None from this file.** `data.py` is bit-parity student → no hidden RG-003 cause here.

RG-003 root causes (per D.26 audit) :
- 🚨 missing `training()` phase (structural, D.28).
- 🚨 missing `create_networks()` 20-copy replication (structural, D.28).
- ❌ config divergences (optimizer, step_size, gamma, n_epochs, `second_order.hidden_dim`).

`data.py` is already paper-faithful.

---

## (g) Open questions (not D.27 blockers)

### (g1) Grammar A terminal reachability
The FSM in port L111-164 has nodes 4 and 5 as "pre-terminal" with path 2 → node 6 (terminal).
Grammar A words therefore almost always end in 'm' (from node 4 or 5 when path=2). Student
code does the same. This is a feature of Dienes (1997) grammars, not a port deviation.

### (g2) Padding convention
Encoded words shorter than 8 letters are zero-padded on the right. `trainer.py:evaluate` uses
`target_second` which relies on `(input == 1)` positions — zero-padded positions correctly
don't count as "active". Consistent.

### (g3) `MAX_WORD_LENGTH` = 8
Paper §A.2 : "strings of 3–8 letters" ✅. Dienes (1997) uses 2-8 — paper extended min to 3.
Port follows paper via `random.randint(3, 8)`.

---

## (h) Summary

- ✅ **All 6 functions + enum + dataclass bit-parity with student.**
- ✅ **8-step math + RNG parity confirmed.**
- ✅ **Legacy quirks preserved** (6-bit chunk with bit-5 unused, grammar-B stalling at node 5).
- ✅ **No new deviation surfaced.**
- ✅ **RG-003 unrelated to data.py** — structural gap lies in `trainer.py` (missing `training()` + 20-network replication).

**D.27 clôturée. 0 code touché. 0 deviations ajoutées. Next : D.28 (RG-003 structural fix).**
