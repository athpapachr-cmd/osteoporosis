import unittest

from fastapi.testclient import TestClient

from main import app


class G3ProductionVisibilityCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def assert_no_store(self, response):
        cache_control = response.headers.get("cache-control", "")
        self.assertIn("no-store", cache_control)
        self.assertIn("no-cache", cache_control)
        self.assertEqual(response.headers.get("pragma"), "no-cache")
        self.assertEqual(response.headers.get("expires"), "0")

    def test_workspace_entrypoint_and_assets_are_not_stale_cacheable(self):
        root = self.client.get("/", follow_redirects=False)
        self.assertEqual(root.status_code, 307)
        self.assertEqual(root.headers.get("location"), "/static/baseline-audit/")
        self.assertIn("no-store", root.headers.get("cache-control", ""))

        for path in (
            "/static/baseline-audit/",
            "/static/baseline-audit/app.js",
            "/static/baseline-audit/progressive-guidance-ui.js",
            "/static/baseline-audit/osteoporosis-longitudinal-summary-core.js",
            "/static/baseline-audit/progressive-guidance.css",
        ):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assert_no_store(response)

    def test_served_bootstrap_contains_g3_visibility_runtime(self):
        app_js = self.client.get("/static/baseline-audit/app.js")
        self.assertEqual(app_js.status_code, 200)
        self.assertIn("osteoporosis-longitudinal-summary-core.js", app_js.text)
        self.assertIn("progressive-guidance-ui.js", app_js.text)

        ui_js = self.client.get("/static/baseline-audit/progressive-guidance-ui.js")
        self.assertEqual(ui_js.status_code, 200)
        self.assertIn('title.textContent = "Σύνοψη ασθενούς"', ui_js.text)
        self.assertIn('badge.textContent = "Νέο"', ui_js.text)
        self.assertIn("patientLongitudinalSummary", ui_js.text)
        self.assertIn("is-newly-surfaced", ui_js.text)


if __name__ == "__main__":
    unittest.main()
