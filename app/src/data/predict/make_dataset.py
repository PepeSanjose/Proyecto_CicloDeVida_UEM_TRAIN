import pandas as pd
from app.src.features.feature_engineering import feature_engineering
from app import init_cols
from app.src.utils.utils import load_model_config
import pickle


def make_dataset(data, artifacts_path):
    """
    Función que permite crear el dataset usado para el entrenamiento
    del modelo.

    Args:
       data (List):  Lista con la observación llegada por request.
       artifacts_path (str):  Ruta local a los artefactos del modelo

    Returns:
       DataFrame. Dataset a inferir.
    """
    try:
        model_info = load_model_config()
        print("---> Getting data")
        data_df = get_raw_data_from_request(data)
        print("---> Transforming data")
        data_df = transform_data(data_df, artifacts_path, model_info["cols_to_remove"])
        print("---> Feature engineering")
        data_df = feature_engineering(data_df)
        print("---> Preparing data for training")
        data_df = pre_train_data_prep(data_df, artifacts_path)

        return data_df.copy()
    except Exception as e:
        print(f"Error in make_dataset: {str(e)}")
        return pd.DataFrame()


def get_raw_data_from_request(data):
    """
    Función para obtener nuevas observaciones desde request

    Args:
       data (List):  Lista con la observación llegada por request.

    Returns:
       DataFrame. Dataset con los datos de entrada.
    """
    try:
        return pd.DataFrame(data, columns=init_cols)
    except Exception as e:
        print(f"Error in get_raw_data_from_request: {str(e)}")
        return pd.DataFrame()


def transform_data(data_df, artifacts_path, cols_to_remove):
    """
    Función que permite realizar las primeras tareas de transformación
    de los datos de entrada.

    Args:
        data_df (DataFrame):  Dataset de entrada.
        artifacts_path (str):  Ruta local a los artefactos del modelo
        cols_to_remove (list): Columnas a retirar.

    Returns:
       DataFrame. Dataset transformado.
    """
    try:
        # ... código de transformación ...

        return data_df.copy()
    except Exception as e:
        print(f"Error in transform_data: {str(e)}")
        return pd.DataFrame()


def pre_train_data_prep(data_df, artifacts_path):
    """
    Función que realiza las últimas transformaciones sobre los datos
    antes del entrenamiento (imputación de nulos)

    Args:
        data_df (DataFrame):  Dataset de entrada.
        artifacts_path (str):  Ruta local a los artefactos del modelo.

    Returns:
        DataFrame. Datasets de salida.
    """
    try:
        # ... código de preparación de datos antes del entrenamiento ...

        return data_df.copy()
    except Exception as e:
        print(f"Error in pre_train_data_prep: {str(e)}")
        return pd.DataFrame()


def input_missing_values(data_df, artifacts_path):
    """
    Función para la imputación de nulos

    Args:
        data_df (DataFrame):  Dataset de entrada.
        artifacts_path (str):  Ruta local a los artefactos del modelo

    Returns:
        DataFrame. Datasets de salida.
    """
    try:
        # ... código de imputación de valores nulos ...

        return data_df.copy()
    except Exception as e:
        print(f"Error in input_missing_values: {str(e)}")
        return pd.DataFrame()


def remove_unwanted_columns(df, cols_to_remove):
    """
    Función para quitar variables innecesarias

    Args:
       df (DataFrame):  Dataset.
       cols_to_remove: List(str). Columnas a eliminar.

    Returns:
       DataFrame. Dataset.
    """
    try:
        return df.drop(columns=cols_to_remove)
    except Exception as e:
        print(f"Error in remove_unwanted_columns: {str(e)}")
        return pd.DataFrame()
