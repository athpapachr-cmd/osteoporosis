from pathlib import Path

BRANCH = "feat/clinical-learning-l1-challenge-foundation-mvp-2026-09-07"
RUNTIME_HEAD = "b53a39f12363e08a7be952e1d7a25fbb5511bca8"
GATE_RUN = "34156659908"
BASE_MAIN = "5f7749c70c6bb3f36fcfc765088d4d363a6bb1d6"

# CURRENT_OPERATIONAL.md
p = Path("CURRENT_OPERATIONAL.md")
s = p.read_text(encoding="utf-8")ns = s.replace(
    "> **STATUS:** CLINICAL LEARNING HUB L-1 — PRODUCT-OWNER AUTHORIZED / IMPLEMENTATION ACTIVE",
    "> **STATUS:** CLINICAL LEARNING HUB L-1 — IMPLEMENTED / TESTED EXACT-HEAD / FINAL REVIEW HANDOFF",
    1,
)
s = s.replace(
    "> **ACTIVE RUNTIME WRITER/LOCK:** ChatGPT — L-1 `clinical_learning/`, `static/clinical-learning/`, bounded `main.py` composition, L-1 tests/workflow and canonical lifecycle updates only.",
    "> **ACTIVE RUNTIME WRITER/LOCK:** NONE — implementation paused for conversation handoff; the next session must fresh-bootstrap and explicitly claim the same bounded L-1 scope before mutation.",
    1,
)
s = s.replace(
    "> **ACTIVE PHYSIOTHERAPY/CU-1 WRITER:** NONE for this session; product owner reports the other conversation is paused until this L-1 step is complete.",
    "> **PHYSIOTHERAPY/CU-1 STATUS:** PAUSED by product owner until this L-1 checkpoint is completed; L-1 must continue to avoid all physio/CU-1/RF owners.",
    1,
)
s = s.replace(
    "# 6. Current lifecycle\n\n```text\nL-0 CONTRACT                         FROZEN / COMPLETE / MERGED\nL-1 PRODUCT-OWNER AUTHORITY          GRANTED\nL-1 SLICE                            ACTIVE\nL-1 IMPLEMENTED                      NO\nL-1 TESTED                           NO\nL-1 EXACT-HEAD REVIEW                NO\nL-1 PR                               NONE\nL-1 MERGED                           NO\nL-1 DEPLOYED                         NO\nL-1 PRODUCTION-SMOKE-VERIFIED        NO\n```",
    f"# 6. Current lifecycle\n\n```text\nL-0 CONTRACT                         FROZEN / COMPLETE / MERGED\nL-1 PRODUCT-OWNER AUTHORITY          GRANTED\nL-1 SLICE                            ACTIVE\nL-1 CORE IMPLEMENTED                 YES\nL-1 CLINICIAN-FACING UX HARDENING    IMPLEMENTED\nL-1 EXACT RUNTIME/UX HEAD            {RUNTIME_HEAD}\nL-1 REGRESSION GATE                  PASS — run {GATE_RUN}\nL-1 PHYSIO/RF ISOLATION              PASS\nL-1 FINAL EXACT-HEAD REVIEW          PENDING\nL-1 PR                               NONE\nL-1 MERGED                           NO\nL-1 DEPLOYED                         NO\nL-1 PRODUCTION-SMOKE-VERIFIED        NO\n```",
    1,
)
start = s.index("# 7. Exact next action")
s = s[:start] + f'''# 7. Exact next action / conversation handoff\n\n```text\nfresh-bootstrap current main + {BRANCH}\n→ verify branch has not moved unexpectedly\n→ claim the bounded L-1 writer lock\n→ perform final exact-head code/product/security/privacy review\n→ specifically review clinician-facing Challenge revision workflow + structured Foundation assessment UX\n→ if any material finding: correct only within frozen L-1 scope and rerun full gate\n→ if clean: canonical closeout to IMPLEMENTED / TESTED / REVIEWED\n→ open bounded L-1 PR to main\n→ HOLD for separate product-owner merge/release authority\n```\n\nKnown exact evidence before handoff:\n\n```text\nmain/base            {BASE_MAIN}\nruntime/UX head      {RUNTIME_HEAD}\nL-1 regression gate {GATE_RUN} — SUCCESS\n```\n\nDo not start PR-1, PR-2, Daily Case Review, Signal work, physiotherapy/CU-1 or RF changes during this closeout.\n'''
p.write_text(s, encoding="utf-8")

# SLICE_PLAN_CURRENT.md
p = Path("SLICE_PLAN_CURRENT.md")
s = p.read_text(encoding="utf-8")
s = s.replace(
    "> **STATUS:** PRODUCT-OWNER AUTHORIZED / IMPLEMENTATION ACTIVE",
    "> **STATUS:** IMPLEMENTED / TESTED EXACT-HEAD CANDIDATE — FINAL REVIEW HANDOFF",
    1,
)
s = s.replace(
    "# 17. Lifecycle / next gate\n\n```text\nL-0 CONTRACT                         FROZEN / COMPLETE / MERGED\nL-1 PRODUCT-OWNER AUTHORITY          GRANTED\nL-1 SLICE DESIGN                     FROZEN BY THIS FILE\nL-1 IMPLEMENTATION                   ACTIVE\nL-1 TESTED                           NO\nL-1 EXACT-HEAD REVIEW                NO\nL-1 PR                               NONE\nL-1 MERGED                           NO\nL-1 DEPLOYED                         NO\nL-1 PRODUCTION-SMOKE-VERIFIED        NO\n```\n\nExact next action:\n\n```text\nclaim CURRENT_OPERATIONAL writer lock\n→ implement only frozen owners/seams\n→ run complete L-1 + inherited L-0 regression gate\n→ exact-head scope/security/privacy review\n→ canonical closeout to IMPLEMENTED / TESTED or REPLAN\n→ HOLD for separate release/merge authority\n```",
    f"# 17. Lifecycle / next gate\n\n```text\nL-0 CONTRACT                         FROZEN / COMPLETE / MERGED\nL-1 PRODUCT-OWNER AUTHORITY          GRANTED\nL-1 SLICE DESIGN                     FROZEN BY THIS FILE\nL-1 CORE IMPLEMENTATION              COMPLETE\nL-1 CLINICIAN-FACING UX HARDENING    COMPLETE\nL-1 EXACT RUNTIME/UX HEAD            {RUNTIME_HEAD}\nL-1 TESTED                           YES — run {GATE_RUN} SUCCESS\nL-1 ADJACENT-OWNER ISOLATION         PASS\nL-1 FINAL EXACT-HEAD REVIEW          PENDING\nL-1 PR                               NONE\nL-1 MERGED                           NO\nL-1 DEPLOYED                         NO\nL-1 PRODUCTION-SMOKE-VERIFIED        NO\n```\n\nExact next action:\n\n```text\nfresh-bootstrap + claim CURRENT_OPERATIONAL writer lock\n→ final exact-head code/product/security/privacy review\n→ verify clinician-facing revision + Foundation assessment UX\n→ fix bounded findings only if needed\n→ rerun full L-1 + inherited L-0 regression gate after any code change\n→ canonical closeout\n→ open bounded L-1 PR\n→ HOLD for separate release/merge authority\n```",
    1,
)
p.write_text(s, encoding="utf-8")

# TODO.md
p = Path("TODO.md")
s = p.read_text(encoding="utf-8")
s = s.replace(
    "- [ ] L-1: protected Challenge JSON import/history + Foundation Map skeleton + spaced-repetition due state.",
    f"- [ ] L-1: implementation + clinician-facing UX are complete/tested on `{RUNTIME_HEAD}`; regression gate `{GATE_RUN}` PASS; final exact-head review, PR, merge/deploy and production smoke remain pending.",
    1,
)
p.write_text(s, encoding="utf-8")

# osteoporosis-change-log.md — append only
p = Path("osteoporosis-change-log.md")
s = p.read_text(encoding="utf-8")
marker = "## 2026-09-07 — Clinical Learning L-1 implementation exact-head regression checkpoint"
if marker not in s:
    s = s.rstrip() + f'''\n\n---\n\n{marker}\n\nOn branch `{BRANCH}`, the bounded L-1 Clinical Learning runtime was implemented for protected Challenge import/history/revisions/delete, reference-verification overlay, the 14-node Osteoporosis Foundation Map, explicit Foundation assessments/state, reviewed due-state scheduling and the first protected clinician-facing Learning Hub UI.\n\nHardening corrected timezone-aware UTC normalization and prevented an older Foundation assessment from overwriting a newer materialized state. Clinician-facing UX was then improved for Challenge revision flow and structured Foundation assessment.\n\nExact runtime/UX head:\n\n```text\n{RUNTIME_HEAD}\n```\n\nFull `Clinical Learning L1 regression gate`:\n\n```text\nrun {GATE_RUN}\nSUCCESS\n```\n\nThe gate covered Python/JavaScript syntax, L-1 runtime/hardening tests, inherited L-0 contract regressions, adjacent-owner isolation and diff hygiene. Physiotherapy/CU-1 and RF owners were not mutated.\n\nLifecycle at this checkpoint:\n\n```text\nL-1 CORE IMPLEMENTED              YES\nL-1 CLINICIAN-FACING UX           IMPLEMENTED\nL-1 EXACT-HEAD REGRESSION GATE    PASS\nFINAL EXACT-HEAD REVIEW           PENDING\nPR                                NONE\nMERGED                            NO\nDEPLOYED                          NO\n```\n\nThe product owner paused the separate physiotherapy workstream until this L-1 checkpoint is completed. The next legitimate action is final exact-head review, then canonical closeout and a bounded PR/release HOLD if clean.\n'''
p.write_text(s, encoding="utf-8")
