from pathlib import Path
import re

BRANCH = "design/clinical-learning-l0-contract-freeze-2026-09-07"
SUBSTANTIVE_HEAD = "afaf9d5d0c7df9d53f8ec714f74d7b44e3a69f96"
SUBSTANTIVE_RUN = "34147429373"

# SLICE_PLAN_CURRENT.md
p = Path("SLICE_PLAN_CURRENT.md")
s = p.read_text(encoding="utf-8")
s = s.replace(
    "> **STATUS:** CONTRACT CANDIDATE COMPLETE / MACHINE GATE PASS / ACTIVE-WRITER EXACT DESIGN REVIEW PASS / INDEPENDENT REVIEW PENDING",
    "> **STATUS:** INDEPENDENT CLOSURE PASS / CONTRACT FREEZE CLOSEOUT ACTIVE / FINAL EXACT-HEAD GATE PENDING",
    1,
)
s = s.replace(
    "> **Substantive exact tested/reviewed contract head:** `76d5fba68a3c4f289fe0d8438fbb1be3f5689a09`.",
    f"> **Substantive corrected contract head:** `{SUBSTANTIVE_HEAD}`.",
    1,
)
s = s.replace(
    "> **Machine evidence:** `Clinical Learning L0 contract gate`, run `34144552849` — SUCCESS.",
    f"> **Substantive machine evidence:** `Clinical Learning L0 contract gate`, run `{SUBSTANTIVE_RUN}` — SUCCESS.",
    1,
)
s = s.replace(
    "> **Exact design review:** `CLINICAL_LEARNING_L0_DESIGN_REVIEW_V1.md` — active-writer PASS, explicitly NOT independent.",
    "> **Active-writer design review:** `CLINICAL_LEARNING_L0_DESIGN_REVIEW_V1.md` — PASS, explicitly NOT independent.\n> **Independent review:** `CLINICAL_LEARNING_L0_INDEPENDENT_REVIEW_V1.md` — CLOSURE PASS / material open finding NONE on the corrected substantive contract head.",
    1,
)

start = s.index("# 18. Acceptance evidence so far")
end = s.index("\n# 20. Out of scope / deferred", start)
replacement = f'''# 18. Acceptance evidence / independent closure review

Corrected substantive contract head:

```text
{SUBSTANTIVE_HEAD}
```

Workflow:

```text
Clinical Learning L0 contract gate
run {SUBSTANTIVE_RUN}
SUCCESS
```

Independent review:

```text
CLINICAL_LEARNING_L0_INDEPENDENT_REVIEW_V1.md
CLOSURE PASS
MATERIAL OPEN FINDING NONE
```

The original pre-independent-review candidate did not receive an automatic pass. Independent review found material but bounded L-0 contract defects; they were corrected within the existing design authority and the full gate was rerun successfully before the CLOSURE PASS.

The corrected contract/gate now covers:

```text
YAML/object/reference integrity                         PASS
Foundation graph integrity                              PASS
Fact Ledger / supersession / disclosure references      PASS
immutable revision/idempotency/conflict semantics       PASS
external-import server authority                        PASS
reusable reference-verification overlay                 PASS
content purge + non-content tombstone                    PASS
due-item source provenance / deletion integrity          PASS
unknown-field rejection + PHI/privacy boundary           PASS
sanitized errors/logging                                 PASS
bibliographic PMID/DOI/URL numeric exclusions            PASS
Foundation transition/evidence authority                 PASS
Daily Case Review eligible-only + revision integrity     PASS
Signal authority separation                              PASS
deferred/completed due occurrence semantics              PASS
baseline shadow methodology                              PASS
L-1 owner/exclusion boundary                             PASS
design-only scope + diff hygiene                         PASS
```

---

# 19. Findings corrected across L-0 reviews

Active-writer review corrected bounded issues including ineligible Daily Case Review persistence, bibliographic-number privacy false positives, Challenge tombstone semantics, external-import authority, topic/ontology ownership, due occurrence semantics and premature Challenge-to-Foundation coupling.

Independent review then found and closed additional material contract defects:

```text
reference verification vs immutable revision authority
→ reusable server/clinician-owned verification overlay

Challenge deletion vs nested learning-action due rows
→ source-artifact provenance + source/target cleanup

incomplete PHI scan surface / unknown imported keys
→ recursive unknown-field rejection + all persistable untrusted strings scanned

privacy rejection could echo rejected content
→ sanitized field-path/code errors and non-content routine logs

internal cross-object reference corruption paths
→ fail-closed uniqueness/resolution/integrity constraints

DailyCaseReview revision predecessor ambiguity
→ explicit immutable predecessor/revision semantics

embedded linked_signal_ids could duplicate Signal authority
→ future shared Signal engine remains dynamic authority

source_artifact_deleted conflicted with append-only Foundation attempts
→ deletion state resolved externally without rewriting reviewed attempts

deferred due state lacked deterministic reactivation
→ future defer remains deferred; today due; past overdue; completed occurrence terminal
```

Independent disposition: **CLOSURE PASS / material open finding NONE**.
'''
s = s[:start] + replacement + s[end:]

start = s.index("# 21. Lifecycle / exact next gate")
s = s[:start] + f'''# 21. Lifecycle / exact next gate

```text
PRODUCT DIRECTION                    APPROVED
L-0 FIELD-LEVEL CONTRACT             CANDIDATE COMPLETE
SUBSTANTIVE MACHINE GATE             PASS — run {SUBSTANTIVE_RUN}
ACTIVE-WRITER EXACT DESIGN REVIEW    PASS
INDEPENDENT EXACT-HEAD REVIEW        CLOSURE PASS
MATERIAL OPEN FINDING                NONE
L-0 CONTRACT FROZEN / COMPLETE       NO — final canonical exact-head gate pending
L-0 MERGED TO MAIN                   NO
L-1 RUNTIME AUTHORITY                NONE
```

Exact next action:

```text
canonical closeout
→ remove temporary closeout helpers
→ final exact-head L-0 gate
→ if PASS, mark L-0 CONTRACT FROZEN / COMPLETE
→ open bounded design PR to main
→ RELEASE / DESIGN HOLD
→ HOLD for separate product-owner L-1 runtime implementation decision
```

Any material contract change after the independently reviewed substantive head invalidates the CLOSURE PASS and requires another substantive review. Status/documentation-only closeout changes do not authorize L-1 implementation.
'''
p.write_text(s, encoding="utf-8")

# CURRENT_OPERATIONAL.md
p = Path("CURRENT_OPERATIONAL.md")
s = p.read_text(encoding="utf-8")
s = re.sub(
    r"> \*\*STATUS:\*\*.*",
    "> **STATUS:** CLINICAL LEARNING HUB L-0 — INDEPENDENT CLOSURE PASS / CANONICAL CLOSEOUT ACTIVE / FINAL EXACT-HEAD GATE PENDING",
    s,
    count=1,
)
s = re.sub(
    r"> \*\*Substantive exact tested contract head:\*\*.*",
    f"> **Corrected substantive contract head:** `{SUBSTANTIVE_HEAD}`.",
    s,
    count=1,
)
s = re.sub(
    r"> \*\*Machine evidence:\*\*.*",
    f"> **Substantive machine evidence:** `Clinical Learning L0 contract gate`, run `{SUBSTANTIVE_RUN}` — SUCCESS.",
    s,
    count=1,
)
s = re.sub(
    r"> \*\*Exact design review:\*\*.*",
    "> **Active-writer review:** `CLINICAL_LEARNING_L0_DESIGN_REVIEW_V1.md` — PASS, not independent.\n> **Independent review:** `CLINICAL_LEARNING_L0_INDEPENDENT_REVIEW_V1.md` — CLOSURE PASS / material open finding NONE.",
    s,
    count=1,
)
s = s.replace(
    "> **ACTIVE DESIGN WRITER/LOCK:** NONE after canonical closeout; branch is review-only pending independent review.",
    "> **ACTIVE DESIGN/CANONICAL WRITER:** ChatGPT — canonical closeout only; no material contract mutation authorized without re-review.",
    1,
)
s = s.replace(
    "> **L-0 merge authority:** HOLD pending independent exact-head review.",
    "> **L-0 merge authority:** NONE in this step; bounded design PR may be opened after final exact-head gate, then HOLD.",
    1,
)

# Replace automated evidence section and downstream lifecycle/next action while preserving privacy sections.
start = s.index("# 5. Automated evidence")
end = s.index("\n# 7. Lifecycle", start)
section = f'''# 5. Independent closure evidence

Corrected substantive contract head:

```text
{SUBSTANTIVE_HEAD}
```

Machine gate:

```text
Clinical Learning L0 contract gate
run {SUBSTANTIVE_RUN}
SUCCESS
```

Independent review:

```text
CLINICAL_LEARNING_L0_INDEPENDENT_REVIEW_V1.md
CLOSURE PASS
MATERIAL OPEN FINDING NONE
```

The independent review did not adopt the active-writer review as its conclusion. It identified material bounded contract defects, corrected them within L-0 design authority, and reran the complete machine gate before issuing CLOSURE PASS.

Key corrected ownership/integrity boundaries include reusable reference-verification overlay ownership, due-item source provenance for Challenge deletion, recursive unknown-field/PHI validation with sanitized rejection, fail-closed internal references, Daily Case immutable revision semantics, shared Signal-engine authority, append-only Foundation assessment evidence, and deterministic deferred due-state reactivation.

---

# 6. Final closeout gate still required

The independent substantive review is complete, but L-0 is not yet declared COMPLETE until the status/canonical closeout commits themselves pass the same exact-head contract gate.

No material contract change may be introduced during closeout. If one is needed, the independent CLOSURE PASS must be reopened.
'''
s = s[:start] + section + s[end:]

start = s.index("# 7. Lifecycle")
s = s[:start] + f'''# 7. Lifecycle

```text
PRODUCT DIRECTION                    APPROVED
L-0 FIELD-LEVEL CONTRACT             CANDIDATE COMPLETE
SUBSTANTIVE CONTRACT GATE            PASS — run {SUBSTANTIVE_RUN}
ACTIVE-WRITER EXACT DESIGN REVIEW    PASS
INDEPENDENT EXACT-HEAD REVIEW        CLOSURE PASS
MATERIAL OPEN FINDING                NONE
CANONICAL CLOSEOUT                   IN PROGRESS
FINAL EXACT-HEAD GATE                PENDING
L-0 CONTRACT FROZEN / COMPLETE       NO — final gate pending
L-0 MERGED TO MAIN                   NO
L-1 RUNTIME IMPLEMENTATION           NOT AUTHORIZED
```

---

# 8. Exact next action / HOLD

```text
finish canonical closeout
→ remove temporary closeout helpers
→ run final exact-head L-0 gate
→ if PASS: mark L-0 CONTRACT FROZEN / COMPLETE
→ open bounded design PR to main
→ RELEASE / DESIGN HOLD
→ HOLD for separate product-owner L-1 implementation authority
```

Forbidden:

```text
NO L-1 runtime/database implementation
NO learning API runtime routes
NO Learning Hub production UI
NO external learning credential
NO Daily Case Review runtime
NO raw transcript persistence
NO patient-record mutation
NO Signal promotion
NO background cron
NO production config mutation
```
'''
p.write_text(s, encoding="utf-8")

# TODO.md — durable progress without claiming final exact-head gate yet.
p = Path("TODO.md")
s = p.read_text(encoding="utf-8")
s = s.replace(
    "## 1.10A Clinical Learning Hub — L-0 DESIGN ACTIVE",
    "## 1.10A Clinical Learning Hub — L-0 INDEPENDENT CLOSURE PASS / FINAL GATE PENDING",
    1,
)
s = s.replace(
    "- [ ] L-0: freeze implementable learning object, provenance, privacy, due-state, revision, persistence and Signal contracts.\n- [ ] L-0: identify exact L-1 API/database/UI owners and complete independent design review.",
    "- [x] L-0: freeze implementable learning object, provenance, privacy, due-state, revision, persistence and Signal contracts on corrected substantive head `afaf9d5d0c7df9d53f8ec714f74d7b44e3a69f96`.\n- [x] L-0: identify exact L-1 API/database/UI owners and complete independent design review with CLOSURE PASS / material open finding none.\n- [ ] L-0: complete canonical closeout + final exact-head gate + bounded design PR/HOLD before declaring merge-ready lifecycle closure.",
    1,
)
p.write_text(s, encoding="utf-8")

# Append-only changelog.
p = Path("osteoporosis-change-log.md")
s = p.read_text(encoding="utf-8")
marker = "## 2026-09-07 — Clinical Learning Hub L-0 independent review reached CLOSURE PASS"
if marker not in s:
    s = s.rstrip() + f'''\n\n---\n\n{marker}\n\nAn independent exact-head L-0 review did not accept the prior active-writer PASS as authority. On the original candidate it found material but bounded contract defects involving immutable reference-verification ownership, Challenge-delete/due referential integrity, PHI scan coverage and sanitized rejection, internal cross-object references, Daily Case revision semantics, Signal authority, Foundation source-deletion semantics and deferred due-state reactivation.\n\nThose findings were corrected within the granted L-0 design authority. The corrected substantive contract head is:\n\n```text\n{SUBSTANTIVE_HEAD}\n```\n\nThe full `Clinical Learning L0 contract gate` then passed at run `{SUBSTANTIVE_RUN}`. `CLINICAL_LEARNING_L0_INDEPENDENT_REVIEW_V1.md` records the independent disposition:\n\n```text\nCLOSURE PASS\nMATERIAL OPEN FINDING NONE\nL-0 CONTRACT ELIGIBLE TO FREEZE / COMPLETE\nL-1 IMPLEMENTATION NOT AUTHORIZED\n```\n\nCanonical closeout and a final exact-head gate remain required before L-0 is declared complete and placed in design/release HOLD. No runtime/database/API/UI implementation, patient-data mutation, raw-transcript persistence, production configuration or RF mutation occurred in this review.\n'''
p.write_text(s, encoding="utf-8")
