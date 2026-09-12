# Commercial Products — Physiotherapy Referral

This subtree owns **commercial/product truth** for the clinician-facing Physiotherapy Referral product that is implemented inside the Clinical Excellence Cockpit.

Bootstrap order for product work:

1. read the six root canonicals;
2. read `commercial_products/physio_referral/CURRENT.md`;
3. read `commercial_products/physio_referral/PRODUCT_CONTEXT_CURRENT.md`;
4. read `commercial_products/physio_referral/PRODUCT_PLAN.md` when roadmap/commercial context is needed;
5. inspect technical implementation only after that under `clinic_utilities/physio_referral_product/` and the protected Cockpit route.

Authority split:

```text
root canonicals
→ repo-wide operations / writer / release truth

commercial_products/physio_referral/
→ product philosophy / commercial strategy / product lifecycle / market context

clinic_utilities/physio_referral_product/
→ clinical + UX technical contracts / validators / implementation tests

static/clinic-utilities/physio-referral/
+ clinic_utilities/physio_referral_api.py
→ Cockpit production UI/API integration
```

This subtree is not a seventh root canonical authority.

Current diagnosis vertical: **Knee Osteoarthritis only**.
