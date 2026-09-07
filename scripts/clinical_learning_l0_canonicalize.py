from pathlib import Path
import re

# TODO roadmap: close RF and activate Clinical Learning L-0/L-1 roadmap.
p = Path('TODO.md')
s = p.read_text(encoding='utf-8')
s = re.sub(
    r'C1, G-1, G-2 and G-3 are \*\*implemented / tested / merged / deployed / production-smoke-verified\*\*\..*?\n\n---',
    'C1, G-1, G-2 and G-3 are **implemented / tested / merged / deployed / production-smoke-verified**. G-4 workspace ergonomics are also released. Native RF v2 is now **implemented / tested / merged / deployed / production-smoke-verified** through PRs #75, #76 and #77. Online signature automation is deliberately deferred; no signature image is stored in the public repository. RF remains not pilot-validated and is closed for now unless new material evidence appears.\n\n---',
    s,
    count=1,
    flags=re.S,
)

start = s.index('## 1.6 G-4 Workspace ergonomics + RF utility navigation')
end = s.index('\n## 1.7 Heidi-first capture', start)
rf_section = '''## 1.6 G-4 Workspace ergonomics + native RF v2 — PRODUCTION-SMOKE-VERIFIED / CLOSED FOR NOW

- [x] Release G-4 workspace ergonomics through PR #72 and complete workspace smoke.
- [x] Release bounded authenticated RF gateway hotfix through PR #73; later auth/form leg returned `200`.
- [x] Stop obsolete old-form create/PDF smoke after the authoritative RF form changed.
- [x] Replan RF ownership into native Clinical Excellence Clinic Utilities.
- [x] Release native Category-A A.1/A.2 workflow through PR #75.
- [x] Configure confirmed server-side doctor profile and Medikey / DIROS / Thermedico product catalog.
- [x] Release unilateral target, system-derived exact location, 0..3 medication capacity and parser corrections through PR #76.
- [x] Release imaging-attachment semantic guard through PR #77.
- [x] Product-owner production smoke confirms A.1, A.2, unilateral target/location, medication behavior, obvious laboratory-PDF rejection, real/ambiguous imaging workflow and final PDF generation are working as intended.
- [x] Preserve `RF APPLICATION REQUEST != ACTUAL RF PROCEDURE` and RF-specific persistence boundaries.
- [x] Keep signature automation out of the public repository; manual/external signing is accepted for now.
- [ ] Only reopen RF for a material production defect, authoritative form change, safety/data-integrity issue or explicit new workflow requirement.

RF v2 is **production-smoke-verified**. It is not `PILOT-VALIDATED`, and pilot validation is not required to resume the primary Module-01 roadmap.
'''
s = s[:start] + rf_section + s[end:]

if '## 1.10A Clinical Learning Hub' not in s:
    marker = '\n## 1.11 Thirty-case scored system-assisted baseline'
    learning = '''
## 1.10A Clinical Learning Hub — L-0 DESIGN ACTIVE

- [x] Approve Foundation Map + Clinical Challenges + Daily Heidi-backed Real-Case Review as one reusable Core learning architecture.
- [x] Add canonical detailed design: `CLINICAL_LEARNING_HUB_DESIGN_V1.md`.
- [x] Require a Fact Ledger so progressive-disclosure/synthetic facts cannot be mistaken for real patient facts.
- [x] Require clinician self-review before AI critique in Daily Case Review.
- [x] Preserve raw Heidi transcript as ephemeral and reuse PR-1/PR-3 owners rather than create a parallel AI stack.
- [x] Define Foundation states: `FORMAL_SOLID`, `INTUITIVE_UNSTRUCTURED`, `FRAGMENTED`, `UNKNOWN_UNTESTED`.
- [x] Define daily/weekly/periodic learning cadence and baseline-intervention boundary.
- [ ] L-0: freeze implementable learning object, provenance, privacy, due-state, revision, persistence and Signal contracts.
- [ ] L-0: identify exact L-1 API/database/UI owners and complete independent design review.
- [ ] L-1: protected Challenge JSON import/history + Foundation Map skeleton + spaced-repetition due state.
- [ ] PR-1/PR-2/PR-3: establish reusable transcript and Practice Review seams.
- [ ] L-2: activate Daily Case Review using real eligible Heidi-backed encounters.
- [ ] After baseline lock, activate clinician-facing daily coaching as a formal improvement intervention and re-measure.
- [ ] Later: narrow external ingestion capability (`learning.challenge.write`) without broad patient/encounter/RF authority.

Daily visible AI coaching remains an intervention and stays hidden by default during the 30-case scored system-assisted baseline unless methodology is explicitly REPLANned and the cohort relabelled.

'''
    s = s.replace(marker, learning + marker, 1)

s = re.sub(
    r'- \[ \] Native RF v2 authenticated end-to-end production smoke[^\n]*',
    '- [x] Native RF v2 authenticated end-to-end production smoke completed through the #76/#77 corrected production path.',
    s,
)
s = s.replace(
    '- [x] Native RF Clinic Utility ownership migration implemented/release-candidate-tested; remaining release lifecycle is tracked in §1.6.',
    '- [x] Native RF Clinic Utility ownership migration released and production-smoke-verified; RF is closed for now under §1.6.',
)

order_start = s.index('# 8. BROAD IMPLEMENTATION ORDER')
code_start = s.index('```text', order_start)
code_end = s.index('```', code_start + len('```text')) + 3
new_order = '''```text
1. C1 authoritative Finish release/smoke — closed
2. G-1 dynamic-guidance mechanics — production-smoke-verified / closed
3. G-2 evidence-backed osteoporosis guidance — production-smoke-verified / closed
4. G-3 guidance salience + longitudinal patient summary — production-smoke-verified / closed
5. G-4 / native RF v2 — production-smoke-verified / closed for now
6. L-0 Clinical Learning Hub contract/design freeze
7. L-1 Challenge import + Foundation MVP
8. PR-1 transcript extraction
9. PR-2 inline provisional population
10. guided card UX sufficient for real use
11. 5-case system-assisted pilot
12. one refinement + contract freeze
13. Quick Practice Review shadow capability
14. 30-case system-assisted scored baseline
15. baseline lock
16. clinician-facing reviewed Signals/interventions + Daily Case Review activation
17. one longitudinal closed improvement loop
18. re-measurement / prompt-dependence trend where valid
19. final Module 01 closure review
20. later breadth/generalization
```'''
s = s[:code_start] + new_order + s[code_end:]
p.write_text(s, encoding='utf-8')

# Detailed design status: canonical direction in main, with L-0 still responsible for exact implementable freeze.
p = Path('CLINICAL_LEARNING_HUB_DESIGN_V1.md')
s = p.read_text(encoding='utf-8')
s = s.replace(
    '> **STATUS:** PRODUCT-OWNER-APPROVED DIRECTION / PRE-IMPLEMENTATION DESIGN INPUT',
    '> **STATUS:** PRODUCT-OWNER-APPROVED CANONICAL DESIGN DIRECTION / L-0 CONTRACT FREEZE ACTIVE',
    1,
)
s = s.replace(
    '> **Runtime authority:** NONE — this document does not start implementation',
    '> **Runtime authority:** NONE — L-0 design/canonicalization only; implementation requires separate authority',
    1,
)
p.write_text(s, encoding='utf-8')

# Phase plan date only; 30A content already ported from approved planning artifact.
p = Path('CLINICAL_EXCELLENCE_PLAN.md')
s = p.read_text(encoding='utf-8').replace('> **UPDATED:** 2026-09-06 Asia/Nicosia.', '> **UPDATED:** 2026-09-07 Asia/Nicosia.', 1)
p.write_text(s, encoding='utf-8')

# Append-only history: RF closeout + Learning Hub design activation.
p = Path('osteoporosis-change-log.md')
s = p.read_text(encoding='utf-8')
marker = '## 2026-09-07 — RF v2 production smoke closed; Clinical Learning Hub L-0 activated'
if marker not in s:
    s = s.rstrip() + '''

---

## 2026-09-07 — RF v2 production smoke closed; Clinical Learning Hub L-0 activated

Native RF v2 reached production through PR #75, correction PR #76 and imaging semantic-guard PR #77. Current production identity after #77 is `1d26195c77e186cff98086283252af2eb499dd17`, Render deploy `dep-daeliaks728c7384f3fg` LIVE.

Product-owner authenticated smoke subsequently confirmed the intended A.1/A.2 flow, unilateral target with system-derived location, 0..3 medication-table semantics and corrected parser behavior, laboratory-PDF rejection, real/poorly extractable imaging fallback with explicit clinician confirmation, and final PDF generation. RF v2 is therefore `PRODUCTION-SMOKE-VERIFIED`, not `PILOT-VALIDATED`, and is closed for now unless new material evidence appears.

The official PDF still contains a clinician-signature area. Automated signature-image storage in the public repository was explicitly rejected. Manual/external signing is accepted for now. Any future online signing requires a separate protected security/e-signature design; a PNG/SVG signature must not become a public version-controlled asset.

The product owner then approved moving the primary roadmap back to the Clinical Excellence learning/capture program. `CLINICAL_LEARNING_HUB_DESIGN_V1.md` was promoted into canonical design direction and slice `CORE-LEARNING-HUB-L0-2026-09-07` activated for contract/design freeze only. L-0 covers Challenge, Fact Ledger, Daily Case Review, Foundation Map, due-state, PHI/provenance, Signal and baseline-intervention contracts. No learning runtime implementation is authorized by this design activation.
'''
p.write_text(s, encoding='utf-8')
