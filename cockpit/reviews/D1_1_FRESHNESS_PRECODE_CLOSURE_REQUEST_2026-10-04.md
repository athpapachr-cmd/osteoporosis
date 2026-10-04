# D1.1 freshness — one independent pre-code correction closure

## Decision and exact target

Decide whether FRESH-01/02/03 from the independent BLOCK are closed by the bounded design correction, with affected cumulative D1.1 semantics/privacy still safe. This is **one** read-only delta + affected cumulative closure under PROCEDURES P5/P5.1. No implementation, tests, old post-code review, PR, merge, deployment or Visit Brief work.

```text
REPOSITORY = athpapachr-cmd/osteoporosis
MAIN = 12be588866aee6543444eee0b435e9064c39bc3f
RECEPTION MAIN = bcfa57e0c1ca7358fb898b393e1dafa7b042c238
ORIGINAL BLOCK TARGET BLOB = a6eedf7b465a55f962321e39d3e6c76d1de5676b
CORRECTED DELTA PATH = cockpit/D1_1_FRESHNESS_REPLAN_DELTA_2026-10-04.md
CORRECTED DELTA BLOB = 00e3b4527a8ccbfc6df1a635494da136ccb9dd33
ORIGINAL FINDINGS = cockpit/reviews/D1_1_FRESHNESS_PRECODE_BLOCK_2026-10-04.md
FROZEN EARLIER DESIGN BLOB = 261ee0e7f2fc8b9ebc59310bb1c5e0517439fe33
OLD POST-CODE REQUEST = HOLD / NO VERDICT
```

Fresh-verify source identities and build the single AGENTS six-canonical manifest, then read PROCEDURES P5/P5.1 and cockpit/CURRENT. Reuse previous review evidence for unchanged Q4 facts; examine only the correction delta, original findings and directly affected Reception source paths needed to dispose FRESH-01/02/03. Expand only for a concrete contradiction. Do not repeat the original whole review.

## Finite closure map

| Finding/question | Sufficient evidence |
|---|---|
| FRESH-01 — past/Previous completeness | Compare old/new design text to the original finding and `CalClient.get_bookings` status contract. Confirm explicit `past` plus `upcoming` paginated source reads, bounded local-day/cross-midnight/future window, UID merge, fail-closed incomplete/error behavior and synthetic `past`→Previous oracle. |
| FRESH-02 — call-hour budget | Confirm a real absolute deadline covers both statuses/all pages, no inherited automatic retries/uncapped Retry-After, explicit page ceiling and no partial cache publication; synthetic slow/429 path and unchanged heavy sync are required future oracles. Source: `cal_client.py` retry/timeout behavior and Reception schedule/sync path. |
| FRESH-03 — log minimization | Confirm the design explicitly avoids inherited raw `resp.text` logging on the new schedule path, including error responses, while retaining only non-identifying operational metadata; synthetic PHI log regression is required. Source: `cal_client.py` `_request`. |
| Affected cumulative | Confirm preservation of protected server-to-server minimal response, no browser secret/phone/link/raw payload, weekly Osteoporosis filter, Nicosia/Previous/Current/Next/overlap semantics, availability-vs-booking boundary and old review HOLD. Reuse settled PRE-01/PRE-02 and prior Q4 evidence; only follow a new contradiction. |

Return immediately when these are disposed or a decisive material BLOCK with its smallest correction is established. `UNKNOWN` is not PASS. The author cannot certify closure.

```text
TARGET CORRECTED DELTA BLOB = 00e3b4527a8ccbfc6df1a635494da136ccb9dd33
VERDICT = PASS | BLOCK
COVERAGE = COMPLETE_FOR_DECLARED_SCOPE | PARTIAL
FRESH-01: ...; FRESH-02: ...; FRESH-03: ...; AFFECTED CUMULATIVE: ...
P0:P1:P2 = ...
MATERIAL FINDINGS = NONE | reachable behavior / violated invariant / exact source / smallest correction
NO ADDITIONAL MATERIAL FINDING = YES | NO
UNREVIEWED / MISSING EVIDENCE = NONE | exact gap
FILES EXAMINED = ...
STOP REASON = findings and affected cumulative disposed | decisive material BLOCK | exact evidence gap
IMPLEMENTATION MAY START = YES | NO
RELEASE AUTHORITY = NO MERGE OR DEPLOY
```
