from app.src.data.predict.make_dataset import make_dataset
from mlflow.sklearn import load_model
from mlflow.artifacts import download_artifacts


def predict_pipeline(data):
    """
    Función para gestionar el pipeline completo de inferencia
    del modelo.

    Args:
        data (List): Lista con los datos de entrada.

    Returns:
        list. Lista con las predicciones hechas.
    """

    try:
        print("------> Loading artifacts from the model in Production from MLFlow")
        artifacts_path = load_artifacts()
        print(artifacts_path)

        # Cargando y transformando los datos de entrada
        data_df = make_dataset(data, artifacts_path)

        print("------> Loading the model object in Production from MLFlow")
        model = load_production_model()

        print("------> Obtaining prediction")
        # Realizando la inferencia con los datos de entrada
        return model.predict(data_df).tolist()

    except Exception as e:
        # Si ocurre una excepción, mostramos el mensaje de error y devolvemos una lista vacía como resultado
        print(f"Error in predict_pipeline: {str(e)}")
        return []


def load_production_model(model_name="titanic_model", stage="Production"):
    """
     Función para cargar el modelo de MLFlow

     Args:
         model_name (str):  Nombre del modelo registrado en MLFlow.
         stage (str): Estado del modelo en MLFlow

    Returns:
        obj. Objeto del modelo.
    """

    return load_model(model_uri=f"models:/{model_name}/{stage}")


def load_artifacts(model_name="titanic_model", stage="Production"):
    """
     Función para cargar el modelo de MLFlow

     Args:
         model_name (str):  Nombre del modelo registrado en MLFlow.
         stage (str): Estado del modelo en MLFlow

    Returns:
        str. Ruta local a los articacts
    """

    return download_artifacts(artifact_uri=f"models:/{model_name}/{stage}")
