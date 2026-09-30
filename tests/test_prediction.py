import importlib.util
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from train import train, FEATURES, ROOT


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        path = Path(cls.temp.name) / "model.pkl"
        cls.model, _, _ = train(model_path=path)
        spec = importlib.util.spec_from_file_location("iris_app", ROOT / "flask-app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.client = module.create_app(path).test_client()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_decimal_inputs_match_saved_pipeline_with_reordered_form(self):
        values = [5.1, 3.5, 1.4, 0.2]
        expected = str(self.model.predict(pd.DataFrame([values], columns=FEATURES))[0])
        payload = dict(reversed(list(zip(FEATURES, map(str, values)))))
        response = self.client.post("/predict", data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(expected, response.get_data(as_text=True))

    def test_missing_malformed_and_nonfinite_inputs_are_rejected(self):
        for invalid in ["", "abc", "nan", "inf", "0", "-1"]:
            with self.subTest(value=invalid):
                payload = dict(zip(FEATURES, ["5.1", "3.5", "1.4", invalid]))
                self.assertEqual(self.client.post("/predict", data=payload).status_code, 400)
        self.assertEqual(self.client.post("/predict", data={}).status_code, 400)

    def test_home_page_renders(self):
        self.assertEqual(self.client.get("/").status_code, 200)


if __name__ == "__main__":
    unittest.main()
