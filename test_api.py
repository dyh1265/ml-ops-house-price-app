import unittest
import os
os.environ["PYTEST_CURRENT_TEST"] = "1"  # prevent Streamlit UI from starting

from fastapi.testclient import TestClient
from src.app import app
from src.predict import predict
from unittest.mock import patch


class TestPredictEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.valid_input = {
            "area": 3.0,
            "bedrooms": 2.0,
            "bathrooms": 1.0,
            "stories": 1.0,
            "mainroad": 0.0,
            "guestroom": 0.0,
            "basement": 0.0,
            "hotwaterheating": 1.0,
            "airconditioning": 1.0,
            "parking": 0.0,
            "prefarea": 0.0,
            "furnishingstatus": 0.0
        }

    def test_post_predict_valid_input(self):
        """Test a valid prediction request."""
        response = self.client.post("/predict", json=self.valid_input)
        self.assertEqual(response.status_code, 200)
        self.assertIn("predicted_price", response.json())
        self.assertIsInstance(response.json()["predicted_price"], float)

    def test_post_predict_negative_area(self):
        """Test prediction with negative area value (Pydantic validation)."""
        invalid_input = self.valid_input.copy()
        invalid_input["area"] = -3.0
        response = self.client.post("/predict", json=invalid_input)
        self.assertEqual(response.status_code, 422)
        self.assertIn("detail", response.json())
        error_detail = response.json()["detail"][0]
        self.assertEqual(error_detail["loc"], ["body", "area"])
        self.assertIn("Input should be greater than or equal to 0", error_detail["msg"])

    def test_post_predict_negative_bedrooms(self):
        """Test prediction with negative bedrooms value (Pydantic validation)."""
        invalid_input = self.valid_input.copy()
        invalid_input["bedrooms"] = -2.0
        response = self.client.post("/predict", json=invalid_input)
        self.assertEqual(response.status_code, 422)
        self.assertIn("detail", response.json())
        error_detail = response.json()["detail"][0]
        self.assertEqual(error_detail["loc"], ["body", "bedrooms"])
        self.assertIn("Input should be greater than or equal to 0", error_detail["msg"])

    def test_post_predict_missing_fields(self):
        """Test prediction with missing required fields."""
        response = self.client.post("/predict", json={
            "area": 3.0,
            "bedrooms": 2.0
        })
        self.assertEqual(response.status_code, 422)
        self.assertIn("detail", response.json())

    def test_post_predict_invalid_data_types(self):
        """Test prediction with invalid data types."""
        invalid_input = self.valid_input.copy()
        invalid_input["area"] = "invalid"
        response = self.client.post("/predict", json=invalid_input)
        self.assertEqual(response.status_code, 422)
        self.assertIn("detail", response.json())

    def test_post_predict_invalid_categorical(self):
        """Test prediction with invalid categorical value."""
        invalid_input = self.valid_input.copy()
        invalid_input["mainroad"] = 2.0
        response = self.client.post("/predict", json=invalid_input)
        self.assertEqual(response.status_code, 422)
        self.assertIn("detail", response.json())
        error_detail = response.json()["detail"][0]
        self.assertEqual(error_detail["loc"], ["body", "mainroad"])
        self.assertIn("Input should be less than or equal to 1", error_detail["msg"])

    def test_post_predict_extreme_values(self):
        """Test prediction with extreme values."""
        extreme_input = {
            "area": 100000.0,
            "bedrooms": 50.0,
            "bathrooms": 20.0,
            "stories": 10.0,
            "mainroad": 1.0,
            "guestroom": 1.0,
            "basement": 1.0,
            "hotwaterheating": 1.0,
            "airconditioning": 1.0,
            "parking": 10.0,
            "prefarea": 1.0,
            "furnishingstatus": 2.0
        }
        response = self.client.post("/predict", json=extreme_input)
        self.assertEqual(response.status_code, 200, f"Unexpected validation error: {response.json()}")
        self.assertIn("predicted_price", response.json())
        self.assertIsInstance(response.json()["predicted_price"], float)

    def test_predict_function_negative_area(self):
        """Test predict function directly with negative area value."""
        invalid_input = self.valid_input.copy()
        invalid_input["area"] = -3.0
        with self.assertRaises(ValueError) as context:
            predict(invalid_input)
        self.assertEqual("Feature 'area' cannot be negative", str(context.exception))

    def test_predict_function_missing_keys(self):
        """Test predict function with missing keys."""
        invalid_input = {"area": 3.0, "bedrooms": 2.0}
        with self.assertRaises(KeyError) as context:
            predict(invalid_input)
        self.assertEqual(str(context.exception).strip("'"), "Missing required input features")
    
    @patch("src.predict.joblib.load")
    def test_post_predict_valid_input_mocked(self, mock_load):
        mock_model = mock_load.return_value
        mock_model.predict.return_value = [123456.78]
        response = self.client.post("/predict", json=self.valid_input)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["predicted_price"], 123456.78)
if __name__ == "__main__":
    unittest.main()