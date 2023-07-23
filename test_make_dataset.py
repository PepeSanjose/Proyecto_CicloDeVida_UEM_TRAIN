import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from app.src.data.predict.make_dataset import make_dataset
from app.src.features.feature_engineering import feature_engineering
from app.src.data.predict.make_dataset import (
    make_dataset,
    get_raw_data_from_request,
    transform_data,
    pre_train_data_prep,
    input_missing_values,
    remove_unwanted_columns
)
from app import init_cols
from app.src.models.predict import load_artifacts
from app.src.utils.utils import load_model_config
import pickle

artifacts_path = load_artifacts()

class TestMakeDataset(unittest.TestCase):
    def setUp(self):
        # Datos de prueba para el dataset
        self.test_data = {
            "column1": [1, 2, 3],
            "column2": [4, 5, 6]
        }
        # Datos de prueba para el DataFrame resultante
        self.expected_data = pd.DataFrame(self.test_data, columns=init_cols)

    def test_get_raw_data_from_request(self):
        # Llamada a la función con datos de prueba
        result = get_raw_data_from_request(self.test_data)

        # Verificación del resultado
        pd.testing.assert_frame_equal(result, self.expected_data)


    @patch("app.src.data.predict.make_dataset.pd.DataFrame")
    @patch("app.src.data.predict.make_dataset.input_missing_values")
    def test_pre_train_data_prep(self, mock_input_missing_values, mock_dataframe):
        # Mock de la función input_missing_values y llamada a la función con datos de prueba
        mock_input_missing_values.return_value = self.expected_data
        result = pre_train_data_prep(self.expected_data, artifacts_path)

        # Verificación del resultado y llamadas a funciones simuladas
        pd.testing.assert_frame_equal(result, self.expected_data)
        mock_input_missing_values.assert_called_once()

    @patch("app.src.data.predict.make_dataset.pd.DataFrame")
    @patch("app.src.data.predict.make_dataset.pd.read_pickle")
    def test_input_missing_values(self, mock_read_pickle, mock_dataframe):
        # Mock de la función read_pickle y llamada a la función con datos de prueba
        mock_read_pickle.return_value = MagicMock(transform=MagicMock(return_value=self.expected_data))
        result = input_missing_values(self.expected_data, artifacts_path)

        # Verificación del resultado y llamadas a funciones simuladas
        pd.testing.assert_frame_equal(result, self.expected_data)
        mock_read_pickle.assert_called_once_with("/path/to/artifacts/imputer.pkl")


    def test_make_dataset(self):
        # Mock de las funciones de transformación y llamada a la función con datos de prueba
        with patch("app.src.data.predict.make_dataset.get_raw_data_from_request") as mock_get_raw_data, \
                patch("app.src.data.predict.make_dataset.transform_data") as mock_transform_data, \
                patch("app.src.data.predict.make_dataset.feature_engineering") as mock_feature_engineering, \
                patch("app.src.data.predict.make_dataset.pre_train_data_prep") as mock_pre_train_data_prep:

            mock_get_raw_data.return_value = self.expected_data
            mock_transform_data.return_value = self.expected_data
            mock_feature_engineering.return_value = self.expected_data
            mock_pre_train_data_prep.return_value = self.expected_data

            result = make_dataset(self.test_data, "/path/to/artifacts")

            # Verificación del resultado y llamadas a funciones simuladas
            pd.testing.assert_frame_equal(result, self.expected_data)
            mock_get_raw_data.assert_called_once()
            mock_transform_data.assert_called_once()
            mock_feature_engineering.assert_called_once()
            mock_pre_train_data_prep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
