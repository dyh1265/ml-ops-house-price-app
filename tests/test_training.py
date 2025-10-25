import unittest
import os
import subprocess
import joblib

class TestTraining(unittest.TestCase):
    def test_training_creates_model_file(self):
        # Remove model.pkl if it exists
        model_path = "model.pkl"
        if os.path.exists(model_path):
            os.remove(model_path)

        # Run the training script
        subprocess.run(["python3", "src/train.py"], check=True)

        # Check if model.pkl was created
        self.assertTrue(os.path.exists(model_path), "model.pkl was not created after training")

        # Load the model and check if it has a predict method
        model = joblib.load(model_path)
        self.assertTrue(hasattr(model, "predict"), "Loaded model does not have a predict method")

if __name__ == "__main__":
    unittest.main()