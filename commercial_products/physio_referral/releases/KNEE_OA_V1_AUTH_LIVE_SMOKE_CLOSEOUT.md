# Knee OA v1 — authenticated live production smoke closeout

> **Production runtime SHA:** `bf527e3836a18491b2758fd293b42f82e0924382`
> **Render deploy:** `dep-daiivp0jo6nc73bl6pug`
> **Production URL:** `https://ortho-reception-backend.onrender.com`
> **GitHub Actions run:** `34681808255`
> **Successful attempt:** `3`
> **Result:** `SUCCESS`
> **Date:** 2026-09-12

## 1. Why this closeout exists

The v4 release closeout had already proven live public assets and the unauthenticated protection boundary, but deliberately did not claim a full authenticated production product smoke because no authorized production credential/session had been used.

The Product Owner subsequently made the existing production `CLINICAL_DATA_KEY` available to GitHub Actions as a protected repository secret with the same name. No production key rotation or application-code change was required.

## 2. Exact successful smoke

Run `34681808255`, attempt `3`, completed `SUCCESS`.

The protected workflow proved against live production:

1. authenticated GET of `/clinical/clinic-utilities/physio-referral` returned the reviewed Knee-OA product surface;
2. authenticated GET of `/clinical/clinic-utilities/physio-referral/api/product/bootstrap` returned the production bootstrap with `synthetic_only=false`, correct deployment context and reviewed defaults;
3. authenticated POST of a generated, non-identifiable right-knee OA state to `/api/product/project` returned an allowed deterministic referral containing right-knee wording and no evidence metadata leakage into referral prose;
4. the same projection boundary with `infection_or_septic_joint_concern` failed closed: export not allowed, blocked true and no referral text;
5. the workflow asserted that only a generated UUID and synthetic/non-identifiable clinical state were supplied.

## 3. Credential and patient-data boundary

The production credential was consumed only through GitHub Actions secret interpolation and was not printed by the workflow.

The smoke did not use:

- patient name;
- identity or GeSY number;
- patient history;
- real referral content;
- session cookie copied from a user browser;
- patient persistence;
- analytics.

## 4. Lifecycle conclusion

For the released v4 production runtime:

```text
MERGED                                      YES
DEPLOYED                                    YES
PUBLIC-ASSET LIVE SMOKE                     PASS
UNAUTHENTICATED AUTH-BOUNDARY LIVE SMOKE    PASS
AUTHENTICATED LIVE PRODUCT SMOKE            PASS
PILOT-VALIDATED                             NO
RECEIVER-VALIDATED                          NO
COMMERCIAL/PAID VALIDATED                   NO
```

This closes the authenticated production lifecycle boundary that previously blocked new Physiotherapy Referral product work.

It does not itself validate receiver usefulness, real-clinic workflow, paid value, iPhone/VoiceOver acceptance, Cyprus/GeSY recommendation fidelity or a second diagnosis.

## 5. Next governance consequence

The planned Cyprus/GeSY OA primary-source audit and jurisdiction-overlay design may proceed.

No local rule becomes active merely because the audit identifies a difference. Any jurisdiction behavior still requires source verification, explicit design review and separate implementation authority.