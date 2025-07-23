import unittest
from src.predict import predict

class TestPredictFunction(unittest.TestCase):
    def test_predict_valid_input(self):
        input_data = {
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

        result = predict(input_data)
        self.assertIsInstance(result, float)

    def test_predict_invalid_input_missing_keys(self):
        input_data = {
            "area": 3.0,
            "bedrooms": 2.0  # Missing other required keys
        }
        with self.assertRaises(KeyError):
            predict(input_data)

    def test_predict_invalid_input_wrong_type(self):
        input_data = {
            "area": "invalid",  # String instead of float
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
        with self.assertRaises(ValueError):
            predict(input_data)


        

if __name__ == "__main__":
    unittest.main()