"""Focused integration tests against the real CU-1 engine, never a safety mock."""
from __future__ import annotations
import copy
import http.client
import json
import sys
import threading
import unittest
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from clinic_utilities.physio_referral_product.prototype import server as p


def request(state=None):
    return {
        "draft_id": str(uuid.uuid4()), "revision": 3, "package_version": p.PACKAGE, "synthetic_only": True,
        "state": state or {"laterality": "right", "formal_assertion_state": "yes", "findings": [],
            "functional_impairments": [], "rehab_directions": list(p.E["default_plan"]["selected"]),
            "adjunct_options": [], "goals": [], "phenotype": {}, "explicit_restrictions": [],
            "clinician_free_text_optional": "", "safety_flags": []}, "dismissed": [],
    }


class AdapterTests(unittest.TestCase):
    def test_real_default_gate_and_no_input_mutation(self):
        req = request(); snapshot = copy.deepcopy(req)
        result = p.project(req)
        self.assertEqual(req, snapshot)
        self.assertTrue(result["gate"]["allowed"])
        self.assertFalse(result["gate"]["blocked"])
        self.assertEqual(result["validation_errors"], [])
        self.assertIn("δεξιού γόνατος", result["text"])
        self.assertEqual(result["suggestions"], [])

    def test_frozen_exact_outputs_through_new_adapter(self):
        primary = p.load_yaml(p.PRODUCT / "contracts/knee_oa_template_fixtures_v1.yaml")["fixtures"]
        edge = p.load_yaml(p.PRODUCT / "contracts/knee_oa_template_edge_fixtures_v1.yaml")["render_fixtures"]
        count = 0
        for fixture in primary + edge:
            with self.subTest(fixture=fixture["id"]):
                state = copy.deepcopy(fixture["input"])
                # This fixture-only UI projection is not accepted clinical input.
                suggested = state.pop("ui_suggested_item_ids", [])
                result = p.project(request(state))
                self.assertTrue(result["gate"]["allowed"], result["validation_errors"])
                self.assertEqual(result["text"], fixture["expected_text"])
                for text in fixture.get("forbidden_substrings", []):
                    self.assertNotIn(text, result["text"])
                for item in suggested:
                    self.assertNotIn(item, result["state"]["rehab_directions"])
                count += 1
        self.assertGreaterEqual(count, 15)
        print(f"Real adapter exact-output fixtures: {count} PASS")

    def test_diagnosis_and_side_are_not_inferred(self):
        for field in ["formal_assertion_state", "laterality"]:
            with self.subTest(field=field):
                req = request(); req["state"][field] = "not_stated"
                result = p.project(req)
                self.assertFalse(result["gate"]["allowed"])
                self.assertIsNone(result["text"])
                self.assertNotIn("Έτοιμη", result["readiness"]["label"])

    def test_real_inherited_safety_flags_block(self):
        for flag in p.FLAGS:
            with self.subTest(flag=flag):
                req = request(); req["state"]["safety_flags"] = [flag]
                result = p.project(req)
                self.assertTrue(result["gate"]["blocked"])
                self.assertFalse(result["gate"]["allowed"])
                self.assertIsNone(result["text"])
                self.assertTrue(any(item["blocked"] for item in result["safety"]))

    def test_unsupported_hidden_and_forged_authority_rejected(self):
        for category, item in [("findings", "true_locking_or_major_mechanical_rom_block"),
            ("findings", "bruising"), ("adjunct_options", "dry_needling"),
            ("rehab_directions", "walking_aid_assessment_and_training"), ("rehab_directions", "weight_management")]:
            with self.subTest(item=item):
                req=request(); req["state"][category].append(item)
                with self.assertRaises(ValueError): p.project(req)
        for field in ["gate", "safety", "normalized_draft", "manual_buffer"]:
            req=request(); req[field]={"allowed":True}
            with self.assertRaises(ValueError): p.project(req)
        req=request();req["state"]["safety"]={"acknowledged_rule_ids":["infection_or_septic_joint_concern"]}
        with self.assertRaises(ValueError):p.project(req)

    def test_types_and_bounds(self):
        mutations = [lambda r:r.update(synthetic_only=False),lambda r:r.update(revision=True),
            lambda r:r.update(package_version="forged"),lambda r:r.update(draft_id="not-a-uuid"),
            lambda r:r["state"].update(findings="pain"),lambda r:r["state"].update(phenotype={"stiffness_symptom":"yes"}),
            lambda r:r["state"].update(clinician_free_text_optional="x"*801)]
        for mutate in mutations:
            req=request();mutate(req)
            with self.assertRaises(ValueError):p.project(req)

    def test_symptoms_do_not_invent_objective_findings_or_treatment(self):
        req=request();req["state"]["phenotype"]={"stiffness_symptom":True,"weakness_symptom_or_context":True}
        result=p.project(req)
        self.assertNotIn("αντικειμενικά",result["text"])
        self.assertNotIn("περιορισμό ενεργητικού",result["text"])
        self.assertNotIn("mobility_exercise_when_restricted",[c["item_id"] for c in result["suggestions"]])

    def test_explicit_suggestion_add_and_stale_rejection(self):
        req=request();req["state"]["functional_impairments"]=["stairs"]
        original=p.project(req)
        candidate=next(c for c in original["suggestions"] if c["item_id"]=="functional_task_retraining")
        self.assertNotIn("λειτουργική επανεκπαίδευση",original["text"])
        self.assertIn("Κλινική προσαρμογή",candidate["source_caption"])
        added=p.project({**req,"candidate":candidate})
        self.assertEqual(added["revision"],req["revision"]+1)
        self.assertIn("λειτουργική επανεκπαίδευση για τις σκάλες",added["text"])
        for change in ["revision","draft_id","package_version","reason_signature"]:
            invalid=copy.deepcopy(req); c=copy.deepcopy(candidate)
            if change=="revision":invalid[change]+=1
            elif change=="draft_id":invalid[change]=str(uuid.uuid4())
            else:c[change]="different"
            with self.assertRaises(ValueError):p.project({**invalid,"candidate":c})
        req["state"]["functional_impairments"]=[]
        with self.assertRaises(ValueError):p.project({**req,"candidate":candidate})

    def test_dismissal_and_new_reason(self):
        req=request();req["state"]["functional_impairments"]=["stairs"]
        candidate=p.project(req)["suggestions"][0]
        req["dismissed"]=[candidate["dismiss_key"]]
        self.assertEqual(p.project(req)["suggestions"],[])
        req["state"]["functional_impairments"].append("sit_to_stand");req["revision"]+=1
        self.assertEqual(len(p.project(req)["suggestions"]),1)

    def test_disagreement_availability_and_greek_copy(self):
        req=request();req["state"]["adjunct_options"]=["acupuncture"]
        result=p.project(req);view=result["evidence"]["acupuncture"]
        self.assertEqual(view["evidence_state"],"guideline_conflict_or_mixed")
        self.assertEqual(len(view["positions"]),5)
        self.assertIn("against",[row["direction"] for row in view["positions"]])
        position_count=0
        for item,v in result["evidence"].items():
            for row in v["positions"]:
                self.assertTrue(row["summary_el"],(item,row["source_id"]))
                self.assertEqual(row["locator_precision"],"source_level")
                position_count+=1
        print(f"Greek source-summary display coverage: {position_count} positions")
        req["availability"]={"NICE_NG226_2022":"source_unavailable"}
        unavailable=p.project(req)
        self.assertEqual(len(unavailable["evidence"]["acupuncture"]["positions"]),5)
        self.assertFalse(unavailable["evidence"]["therapeutic_exercise"]["endorsement"])
        self.assertIn("acupuncture",unavailable["state"]["adjunct_options"])
        self.assertTrue(unavailable["gate"]["allowed"])
        req["state"]["rehab_directions"].remove("therapeutic_exercise")
        self.assertNotIn("therapeutic_exercise",[c["item_id"] for c in p.project(req)["suggestions"]])

    def test_restrictions_and_notes_are_literal_not_authority(self):
        req=request();req["state"]["explicit_restrictions"]=[{"restriction_id":"sport_or_work_restriction","state_or_value":"δοκιμαστικός περιορισμός","source":"clinician_entered"}]
        req["state"]["clinician_free_text_optional"]="  Δοκιμαστική   σημείωση  "
        result=p.project(req)
        self.assertIn("δοκιμαστικός περιορισμός",result["text"])
        self.assertIn("Δοκιμαστική σημείωση",result["text"])
        req["state"]["explicit_restrictions"][0]["state_or_value"]=""
        with self.assertRaises(ValueError):p.project(req)


class HTTPBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=p.ThreadingHTTPServer(("127.0.0.1",0),p.Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def call(self,method,path,body=None,headers=None):
        connection=http.client.HTTPConnection("127.0.0.1",self.server.server_port,timeout=5)
        connection.request(method,path,body,headers or {})
        response=connection.getresponse();result=(response.status,dict(response.getheaders()),response.read())
        connection.close();return result
    def test_static_no_store_and_allowlist(self):
        code,headers,body=self.call("GET","/")
        self.assertEqual(code,200);self.assertEqual(headers["Cache-Control"],"no-store")
        self.assertIn("frame-ancestors 'none'",headers["Content-Security-Policy"])
        self.assertEqual(self.call("GET","/server.py")[0],404)
        self.assertEqual(self.call("GET","/?patient=synthetic")[0],404)
    def test_real_http_projection(self):
        code,_,body=self.call("POST","/api/project",json.dumps(request()),{"Content-Type":"application/json","X-Physio-Prototype":"1"})
        self.assertEqual(code,200);self.assertTrue(json.loads(body)["gate"]["allowed"])
    def test_cross_origin_bad_host_and_missing_header(self):
        common={"Content-Type":"application/json","X-Physio-Prototype":"1"}
        for headers in [{**common,"Origin":"https://untrusted.invalid"},{**common,"Host":"untrusted.invalid"},
                        {"Content-Type":"application/json"},{**common,"Sec-Fetch-Site":"cross-site"}]:
            self.assertEqual(self.call("POST","/api/project",json.dumps(request()),headers)[0],403)
    def test_sanitized_invalid_payload(self):
        req=request();req["state"]["findings"]=["PRIVATE_TEST_MARKER"]
        code,_,body=self.call("POST","/api/project",json.dumps(req),{"Content-Type":"application/json","X-Physio-Prototype":"1"})
        self.assertEqual(code,400);self.assertNotIn(b"PRIVATE_TEST_MARKER",body)
        self.assertEqual(self.call("POST","/api/project","x"*32769,{"Content-Type":"application/json","X-Physio-Prototype":"1"})[0],400)


if __name__=="__main__":unittest.main(verbosity=2)
