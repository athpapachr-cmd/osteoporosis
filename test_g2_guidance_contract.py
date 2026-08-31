from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent

FILES = {
    "evidence": ROOT / "schemas/osteoporosis_guidance_evidence_registry_v1.yaml",
    "rules": ROOT / "schemas/osteoporosis_guidance_rules_v1.yaml",
    "profiles": ROOT / "schemas/osteoporosis_guidance_profiles_v1.yaml",
    "milestones": ROOT / "schemas/osteoporosis_therapy_milestones_v1.yaml",
    "manifest": ROOT / "schemas/osteoporosis_guidance_contract_manifest_v1.yaml",
}

DOMAIN_SET = {
    "fracture_history", "formal_risk", "dxa", "vfa", "secondary_causes",
    "laboratory_monitoring", "falls_function", "sarcopenia", "treatment_history",
    "administrations", "treatment_decision", "transition_safety", "followup_tasks",
    "communication", "understanding", "reflection", "documentation_capture",
}


def load(path):
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def unique(values, label):
    assert len(values) == len(set(values)), f"duplicate {label}: {values}"


def source_claim_index(evidence):
    index = set()
    source_ids = []
    for source in evidence.get("sources", []):
        sid = source["source_id"]
        source_ids.append(sid)
        for claim in source.get("claims", []):
            index.add(f"{sid}#{claim['claim_id']}")
    unique(source_ids, "source_id")
    return set(source_ids), index


def collect_rule_domains(rule):
    domains = []
    if rule.get("domain"):
        domains.append(rule["domain"])
    domains.extend(rule.get("domains", []))
    return domains


def test_contract():
    docs = {name: load(path) for name, path in FILES.items()}
    evidence = docs["evidence"]
    rules_doc = docs["rules"]
    profiles = docs["profiles"]
    milestones = docs["milestones"]
    manifest = docs["manifest"]

    assert evidence["schema"] == "osteoporosis_guidance_evidence_registry_v1"
    assert rules_doc["schema"] == "osteoporosis_guidance_rules_v1"
    assert profiles["schema"] == "osteoporosis_guidance_profiles_v1"
    assert milestones["schema"] == "osteoporosis_therapy_milestones_v1"
    assert manifest["schema"] == "osteoporosis_guidance_contract_manifest_v1"

    source_ids, claim_refs = source_claim_index(evidence)

    rules = rules_doc.get("rules", [])
    rule_ids = [rule["rule_id"] for rule in rules]
    unique(rule_ids, "rule_id")
    rule_id_set = set(rule_ids)

    for rule in rules:
        domains = collect_rule_domains(rule)
        assert domains, f"rule has no target domain: {rule['rule_id']}"
        unknown_domains = set(domains) - DOMAIN_SET
        assert not unknown_domains, f"unknown domains in {rule['rule_id']}: {unknown_domains}"
        refs = rule.get("source_refs", [])
        assert refs, f"clinically active rule has no evidence refs: {rule['rule_id']}"
        for ref in refs:
            assert ref in claim_refs, f"dangling source ref in {rule['rule_id']}: {ref}"

    profile_ids = [p["profile_id"] for p in profiles.get("profiles", [])]
    unique(profile_ids, "profile_id")
    reachable_rules = set()
    for profile in profiles.get("profiles", []):
        for domain in profile.get("product_flow_domains", []):
            assert domain in DOMAIN_SET, f"unknown profile domain {domain} in {profile['profile_id']}"
        for rid in profile.get("conditional_rule_ids", []):
            assert rid in rule_id_set, f"dangling profile rule {rid} in {profile['profile_id']}"
            reachable_rules.add(rid)

    milestone_profile_ids = [p["profile_id"] for p in milestones.get("profiles", [])]
    unique(milestone_profile_ids, "therapy milestone profile_id")
    milestone_ids = []
    for profile in milestones.get("profiles", []):
        for source_id in profile.get("sources", []):
            assert source_id in source_ids, f"unknown milestone source {source_id}"
        for milestone in profile.get("milestones", []):
            milestone_ids.append(milestone["milestone_id"])
            for rid in milestone.get("guidance_rule_ids", []):
                assert rid in rule_id_set, f"dangling milestone rule {rid} in {milestone['milestone_id']}"
                reachable_rules.add(rid)
            for ref in milestone.get("source_refs", []):
                assert ref in claim_refs, f"dangling milestone source ref {ref} in {milestone['milestone_id']}"
            for domain in milestone.get("guidance_domains", []):
                assert domain in DOMAIN_SET, f"unknown milestone domain {domain} in {milestone['milestone_id']}"
    unique(milestone_ids, "milestone_id")

    active_rules = {
        rule["rule_id"] for rule in rules
        if str(rule.get("runtime_state", "")).startswith("active_candidate")
    }
    unreachable = active_rules - reachable_rules
    assert not unreachable, f"active candidate rules unreachable from profiles/milestones: {sorted(unreachable)}"

    forbidden = {item["id"] for item in rules_doc.get("explicit_non_rules", [])}
    assert "CTX_280_OR_300_AUTOMATIC_SECOND_ZOLEDRONATE" in forbidden
    assert "PROLIA_4TH_8TH_10TH_DOSE_GENERIC_MILESTONES" in forbidden

    absent = set(milestones.get("explicitly_absent_milestones", []))
    assert "ctx_280_automatic_second_zoledronate" in absent
    assert "ctx_300_automatic_second_zoledronate" in absent
    assert "prolia_4th_dose_generic_review" in absent

    contracts = manifest.get("contracts", {})
    expected = {
        "evidence_registry": FILES["evidence"],
        "rules_registry": FILES["rules"],
        "visit_profiles": FILES["profiles"],
        "therapy_milestones": FILES["milestones"],
    }
    for key, path in expected.items():
        entry = contracts[key]
        assert (ROOT / entry["path"]).resolve() == path.resolve(), f"manifest path mismatch for {key}"
        assert entry["schema"] == docs[key if key != "visit_profiles" else "profiles"]["schema"] if key != "therapy_milestones" else entry["schema"] == milestones["schema"]

    assert "automatic_zoledronate_from_ctx_280_or_300" in manifest["runtime_activation_boundary"]["prohibited"]
    assert "arbitrary_ordinal_prolia_milestones" in manifest["runtime_activation_boundary"]["prohibited"]


if __name__ == "__main__":
    test_contract()
    print("G2 guidance contract consistency: PASS")
