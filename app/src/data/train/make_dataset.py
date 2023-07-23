import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from app.src.features.feature_engineering import feature_engineering
from app.src.utils.utils import save_object_locally


def make_dataset(path, target, cols_to_remove, model_type="RandomForest"):
    """
    Función que permite crear el dataset usado para el entrenamiento
    del modelo.

    Args:
       path (str): Ruta hacia los datos.
       target (str): Variable dependiente a usar.

    Kwargs:
       model_type (str): tipo de modelo usado.

    Returns:
       DataFrame, DataFrame. Datasets de train y test para el modelo.
    """

    try:
        print("---> Getting data")
        df = get_raw_data_from_local(path)
        print("---> Train / test split")
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=50)
        print("---> Transforming data")
        train_df, test_df = transform_data(train_df, test_df, target, cols_to_remove)
        print("---> Feature engineering")
        train_df = feature_engineering(train_df)
        test_df = feature_engineering(test_df)

        print("---> Preparing data for training")
        train_df, test_df = pre_train_data_prep(train_df, test_df, model_type, target)

        return train_df.copy(), test_df.copy()

    except Exception as e:
        print(f"Error in make_dataset: {str(e)}")


def get_raw_data_from_local(path):
    """
    Función para obtener los datos originales desde local

    Args:
       path (str): Ruta hacia los datos.

    Returns:
       DataFrame. Dataset con los datos de entrada.
    """

    try:
        df = pd.read_csv(path)
        return df.copy()
    except Exception as e:
        print(f"Error in get_raw_data_from_local: {str(e)}")
        return pd.DataFrame()


def transform_data(train_df, test_df, target, cols_to_remove):
    """
    Función que permite realizar las primeras tareas de transformación
    de los datos de entrada.

    Args:
       train_df (DataFrame): Dataset de train.
       test_df (DataFrame): Dataset de test.
       target (str): Variable dependiente a usar.
       cols_to_remove (list): Columnas a retirar.

    Returns:
       DataFrame, DataFrame. Datasets de train y test para el modelo.
    """

    try:
        # ... código de transformación ...

        return train_df.copy(), test_df.copy()

    except Exception as e:
        print(f"Error in transform_data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()


def pre_train_data_prep(train_df, test_df, model_type, target):
    """
    Función que realiza las últimas transformaciones sobre los datos
    antes del entrenamiento (imputación de nulos y escalado)

    Args:
       train_df (DataFrame): Dataset de train.
       test_df (DataFrame): Dataset de test.
       model_type (str): Tipo de modelo a usar.
       target (str): Variable dependiente a usar.

    Returns:
       DataFrame, DataFrame. Datasets de train y test para el modelo.
    """

    try:
        # ... código de preparación de datos antes del entrenamiento ...

        return train_df.copy(), test_df.copy()

    except Exception as e:
        print(f"Error in pre_train_data_prep: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()


def input_missing_values(train_df, test_df):
    """
    Función para la imputación de nulos

    Args:
       train_df (DataFrame): Dataset de train.
       test_df (DataFrame): Dataset de test.

    Returns:
       DataFrame, DataFrame. Datasets de train y test para el modelo.
    """

    try:
        # ... código de imputación de valores nulos ...

        return train_df.copy(), test_df.copy()

    except Exception as e:
        print(f"Error in input_missing_values: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()


# Resto del código sin cambios en las funciones restantes
# ...
