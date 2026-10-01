# PHYSIO P1 EVIDENCE PROVENANCE AUDIT

> **TASK:** PHYSIO-P1-KNEE-OA-REFERENCE-VALIDATION-CORE-BOUNDARY-20261001
> **LANE:** F — material evidence provenance / locator maintenance.
> **STATUS:** COMPLETE — NEEDS LOCATOR MAINTENANCE / NO CLINICAL STATE CHANGE IDENTIFIED.
> **Audit date:** 2026-10-01 Asia/Nicosia.
> **Repository snapshot at audit start:** `1097b8e64a8fd181f33b3e53bf6911ba2296ab9c`.
> **Runtime mutation:** none.
> **Scope:** focused material-claim traceability, not a complete de-novo guideline review.

---

## 1. Audit question

Can a reviewer trace the material Knee-OA evidence behavior from current product semantics to a reviewed source with enough precision to maintain it safely?

The audit sampled the evidence that materially drives:

- the visible starting rehabilitation plan;
- omission/suggestion semantics;
- mixed/conflict badges;
- recommendation-against-routine-use behavior;
- important CY_GESY local differences/status;
- the boundary between international evidence and local clinical/admin policy.

The audit did **not** authorize changes to evidence states, UI, selection defaults, referral prose or runtime.

---

## 2. Current owning artifacts

International evidence authority:

- `clinic_utilities/physio_referral_product/contracts/knee_oa_evidence_contract_v1.yaml`
- `clinic_utilities/physio_referral_product/KNEE_OA_EVIDENCE_DESIGN_V1.md`
- `clinic_utilities/physio_referral_product/KNEE_OA_EVIDENCE_DESIGN_REVIEW_V1.md`

CY_GESY authority:

- `clinic_utilities/physio_referral_product/jurisdictions/CY_GESY/knee_oa_overlay_v1.yaml`
- `commercial_products/physio_referral/jurisdictions/CY_GESY/CYPRUS_GESY_OA_SOURCE_AUDIT_V1.md`

The international source registry currently includes VA/DoD 2026, ACE Singapore 2026, EULAR 2023 update, NICE NG226, AAOS OAK3 and ACR/Arthritis Foundation.

---

## 3. Material sampled claims

| Product claim/state | Source-level verification | Precise locator available from source | Audit result |
|---|---|---|---|
| tailored therapeutic exercise is core | NICE NG226; EULAR 2023 update; ACR/AF; AAOS; VA/DoD | NICE 1.3.1; EULAR recommendation 3; VA/DoD recommendation 5; equivalent named exercise recommendations in ACR/AF and AAOS | wording/state supported |
| strengthening is part of supported exercise, not a uniquely superior universal protocol | NICE NG226; EULAR; ACR/AF; AAOS; VA/DoD | NICE 1.3.1; EULAR recommendation 3; VA/DoD recommendation 6 preserves no preferred PT mode | source-scope handling supported |
| education/self-management belongs in the core plan | EULAR; ACR/AF; VA/DoD; NICE | EULAR recommendation 2; VA/DoD recommendation 4; NICE 1.2/1.3 education-exercise recommendations | wording/state supported |
| manual therapy is internationally mixed and adjunctive | NICE; AAOS; ACR/AF | NICE 1.3.6–1.3.7; AAOS `Manual Therapy` recommendation section; ACR/AF manual-therapy-with-exercise conditional-against recommendation | mixed state supported |
| acupuncture is internationally mixed | NICE; AAOS; ACR/AF; ACE; VA/DoD | NICE 1.3.8; AAOS `Acupuncture` recommendation section; ACE recommendation 6; VA/DoD recommendation 32; ACR/AF acupuncture conditional recommendation | mixed state supported |
| dry needling should not be a selectable routine Knee-OA option | NICE; AAOS; VA/DoD | NICE 1.3.8; AAOS `Dry Needling` consensus section; VA/DoD recommendation 32 | current non-selectable / against-routine-use handling remains defensible |
| CY_GESY local position must not overwrite international state | HIO/OAY OA guideline/adaptation + implementation announcement | local recommendation IDs are stored per position; examples 1.3.1, 1.3.6–1.3.8, 1.3.12–1.3.13 | ownership boundary supported |
| HIO implementation is announced while IT integration remains planned and public PDF metadata remains stale/draft-labelled | HIO/OAY implementation announcement + current hosted OA artifact | announcement dated May 2026; public artifact retains draft metadata | current lifecycle caveat remains correct |

No sampled item required a fresh evidence-state, default-plan or routine-referral wording change.

---

## 4. Finding P1-F001 — international positions lack claim-level locators

### Observation

The international machine contract has good source-level provenance:

- source ID;
- organization/title;
- version/year;
- source-level URL;
- reviewed-on date;
- source direction;
- native strength;
- support scope;
- compact source summary.

However, the individual `positions[]` do **not** carry a structured recommendation/section/page/table locator.

A maintainer can reach the correct source document, but cannot always travel directly from:

~~~text
product item
→ source position
→ exact source recommendation
~~~

without manually searching the source again.

### Disposition

`NEEDS LOCATOR MAINTENANCE`

This is **not** evidence that the sampled clinical direction is wrong. Current primary-source checks support the sampled states and scope distinctions.

### Future bounded maintenance shape

A later docs/contract maintenance slice may add a position-level field such as:

~~~text
source_locator:
  recommendation_id:
  section:
  printed_page:
  stable_anchor:
~~~

The exact schema should be decided in that maintenance slice. P1 does not mutate the evidence contract.

---

## 5. Finding P1-F002 — CY_GESY page-number convention is not normalized across artifacts

### Observation

The CY_GESY overlay is stronger than the international contract for claim-level traceability: local positions carry explicit recommendation identifiers plus `recommendation_page_or_section`.

The companion source audit and overlay sometimes appear one page apart for the same recommendation because one artifact effectively references the PDF viewer/index page while the other uses the page number printed on the PDF.

Example pattern:

~~~text
same recommendation ID
+ same official source
+ page reference differs by one
→ viewer/index page vs printed document page
~~~

The recommendation IDs remain consistent and prevent clinical ambiguity.

### Disposition

`NEEDS LOCATOR MAINTENANCE`

Normalize page semantics explicitly, for example:

~~~text
printed_page
pdf_index_page_optional
recommendation_id
~~~

No local evidence direction, international state, default selection or referral prose change is justified by this finding.

---

## 6. Current-source status check

The focused current-source check found no material contradiction requiring reclassification of the sampled evidence behavior.

In particular:

- the core active-rehabilitation/default-plan rationale remains supported;
- the contract correctly preserves broad-vs-item-specific support scope;
- manual therapy and acupuncture remain appropriate examples of material guideline disagreement;
- the current dry-needling exclusion remains consistent with the reviewed source set;
- the CY_GESY layer continues to separate local clinical guidance from international evidence and from GeSY administrative/reimbursement rules;
- the local lifecycle caveat remains necessary because HIO announced implementation while information-system integration was described as future/planned and the hosted recommendation artifact retained stale draft metadata.

This is a focused maintenance audit, not a claim that every evidence row has been freshly re-reviewed de novo.

---

## 7. Lane-F disposition

~~~text
EVIDENCE STATE / WORDING CHANGE        NOT INDICATED BY SAMPLED AUDIT
DEFAULT PLAN CHANGE                    NOT INDICATED
CY_GESY CORE-STATE MUTATION            NOT INDICATED
RUNTIME CORRECTION                     NOT REQUIRED
INTERNATIONAL CLAIM-LEVEL LOCATORS     MAINTENANCE NEEDED
CY_GESY PAGE-CONVENTION NORMALIZATION  MAINTENANCE NEEDED
BLOCKING EVIDENCE INTEGRITY ISSUE      NONE IDENTIFIED
~~~

Overall Lane F:

**NEEDS LOCATOR MAINTENANCE**

This is non-blocking for continued P1 validation. It should be closed through a future bounded evidence-contract/documentation maintenance action before stronger paid-product evidence-maintenance claims are made.

