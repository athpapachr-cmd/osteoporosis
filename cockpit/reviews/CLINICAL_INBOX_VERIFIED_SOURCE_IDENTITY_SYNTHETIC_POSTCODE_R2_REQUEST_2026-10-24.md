# Clinical Inbox #161 source-verified ADT/referral without DOB — independent POST-CODE R2

**Date:** 2026-10-24 Asia/Nicosia.
**MODE:** ONE fresh independent **READ-ONLY, bounded affected POST-CODE R2**; review the synthetic identity-source implementation only, STOP after PASS/BLOCK/UNKNOWN. Reuse independently PASSed source-identity PRE-CODE PR #210 and all unaffected #208 Clinical Attention contracts. **Do NOT review #209 again**, it has been released and clinician accepted.

**Repo:** `athpapachr-cmd/osteoporosis`. **DRAFT PR:** [#211](https://github.com/athpapachr-cmd/osteoporosis/pull/211).
**Exact FROZEN code/test author HEAD:** `84ced47a701bf7540ee6a83383ecb58d5e512633`.
**Verified production main before #211:** `2b05fecfd23e940e463c98d408cf05a8acc8adc7` (#209 merged/LIVE; docs-only closeout).
**PRE-CODE independently reviewed design:** substantive `50944643cc5c0c40e1e6d698b79269559beddde1`, request `656183f4640ede6a91e46fb5a95ea7c6f29f33a1` in [PR #210](https://github.com/athpapachr-cmd/osteoporosis/pull/210) with independent `PASS | COMPLETE_FOR_DECLARED_SCOPE | P0:P1:P2=0:0:0`, five questions all PASS, [durable receipt](https://github.com/athpapachr-cmd/osteoporosis/blob/a73e5b64aa7e79447e241670ae878943e2582545/cockpit/reviews/CLINICAL_INBOX_SOURCE_IDENTITY_ADT_REFERRAL_PRECODE_R2_PASS_RECEIPT_2026-10-24.md). PR #210 remains DESIGN-ONLY DRAFT/unmerged and is not a live identity provider.
**Product Owner:** reject compulsory DOB, prefer previously verified Cyprus ADT/qualified referral identity and minimum clinician interaction; normal real clinical data activation remains unapproved.

## Exact implemented scope

The implementation is limited to existing authenticated **`/clinical/synthetic-161`** route; no normal Clinical Inbox data, no real patients/identifiers/email, no DDL/grant/provider/SMS/result writer:
- `clinical_attention_phase_a.py`: source-scoped, server-owned **fictional** prior referral and qualified ADT evidence proposal, isolated per-item proof (fake reference not national ID); five protected synthetic items including GESY `gesy_notice` results-available without actual lab values and one name-only candidate. New confirm action requires matching `patient_id + evidence_kind + evidence_ref`, explicit clinician-verified flag, current version CAS/idempotency; DOB field removed, mixed/name-only linking denied. Existing reviewed state invalidated on qualified cross-patient relink, mixed/correction semantics unchanged.
- `clinical_synthetic_161.py`: pass the three existing synthetic patient IDs into the isolated attention store; protected eligible patient read omits DOB without modifying real registry or fake seed DOB.
- `static/cockpit/clinical-inbox.js`: show protected source-verified evidence text and preselected matching synthetic patient, checkbox then confirmation; no DOB or raw ADT entry. No source proof: explicit unlinked/warning and **no link form**. #209 topbar peeks and normal unconnected Inbox unaffected.
- `test_cockpit_attention_phase_a.py` and `test_cockpit_synthetic_161.py`: wrong patient/proof/field/status 409/422, protected fixture authority, candidate-only/mixed, exact idempotency/409/revision, synthetic requalified cross-patient re-review, original task owner untouched, actual Chromium no-DOB link/401/source filters/peek.

**No real GESY identifiers or pasted emails are committed in GitHub.** A fake server "proof" is ONLY a synthetic oracle standing in for future independently qualified evidence, never permission to create a real verified registry.

## Five finite implementation R2 questions

1. **Scope/auth:** Are new identities and proofs confined to the authenticated synthetic #161 test owner, with no new production patient/ADT mappings, schema, migration, mail/provider endpoints, data transfer or external communication? Does an unknown actual patient ID fail closed?
2. **Source evidence strength:** Does a prequalified source match require **both** exact item/document scope and one unique protected patient + evidence-kind/reference + clinician approval, checked SERVER-SIDE under CAS/version/idempotency? Reject different patient, borrowed proof, DOB-only, name-only and mixed PDF. Does source/account document part retain unique identity?
3. **Clinical semantics and correction:** GESY notification explicitly means "results available" and not a completed/refuted test; no lab values/clinical facts/Pending-task resolution. Does source revision/relink require re-review (old patient projection removed, old disposition invalidated) while preserving source audit and preventing stale approval?
4. **Clinician UX/privacy:** At 1280/680/390, is DOB absent from synthetic Inbox confirmation without leaking raw/real ADT, with readable evidence and one explicit confirmation step? Weak/conflicting source stays unlinked; logout, same-tab account switch, late/current 401, modal/DOM purge and #209 Inbox/topbar dropdown UX remain safe.
5. **Bounded evidence:** Read exact frozen code HEAD and relevant protected route/tests/diffs. Existing author GitHub Actions on the exact code/test SHA **11/11 SUCCESS**, including [Visit Brief/#161](https://github.com/athpapachr-cmd/osteoporosis/actions/runs/40114190758) **51+15 PASS**, Cockpit Home 40114190822, Canonical 40114190824, and inherited PHYSIO/Learning owners. No later runtime byte change to review; no new R2 for settled #209/#210.

Return `PASS | BLOCK | UNKNOWN`, `COMPLETE_FOR_DECLARED_SCOPE | PARTIAL`, P0:P1:P2 counts, five dispositions, exact source blobs/HEAD, any reachable material finding with smallest correction, STOP. **No code, PR merge, Render deploy, credential access, patient PHI, real identifier linking or additional review chain.** Independent PASS will close this new bounded post-code gate only, not authorize real registry/provider work or deployment.
