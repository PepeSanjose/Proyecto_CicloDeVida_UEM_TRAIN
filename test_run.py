import unittest
from unittest.mock import patch
from run import app
from app.src.models import train_model

class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def tearDown(self):
        pass

    def test_root(self):
        response = self.app.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"Proyecto": "Pantalla de inicio"})

    @patch("app.train_model.training_pipeline")
    def test_train_model_route_success(self, mock_training_pipeline):
        response = self.app.get("/train-model")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"TRAINING MODEL": "El proyecto se ha entrenado correctamente"})
        mock_training_pipeline.assert_called_once()

    @patch("app.train_model.training_pipeline", side_effect=Exception("Error de entrenamiento"))
    def test_train_model_route_failure(self, mock_training_pipeline):
        response = self.app.get("/train-model")
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json, {"error": "Error de entrenamiento"})
        mock_training_pipeline.assert_called_once()

    def test_predict_route_success(self):
        # Datos de prueba para el método POST
        data = {"key": "value"}

        response = self.app.post("/predict", json=data)
        self.assertEqual(response.status_code, 200)

        # Aquí verificas que la respuesta del servicio sea la esperada en función de tus datos de prueba
        # Por ejemplo, puedes verificar que la respuesta es un diccionario con la clave "Predicted value"
        self.assertIsInstance(response.json, dict)
        self.assertIn("Predicted value", response.json)

    @patch("app.predict_pipeline", side_effect=Exception("Error de inferencia"))
    def test_predict_route_failure(self, mock_predict_pipeline):
        # Datos de prueba para el método POST
        data = {"key": "value"}

        response = self.app.post("/predict", json=data)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json, {"error": "Error de inferencia"})
        mock_predict_pipeline.assert_called_once()

    # Puedes agregar más casos de prueba según tus necesidades

if __name__ == "__main__":
    unittest.main()
