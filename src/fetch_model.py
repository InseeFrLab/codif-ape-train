"""
Fetching models script.
"""

import os

import mlflow


def fetch_model(
    server_uri: str, model_name: str, model_version: str
) -> mlflow.pyfunc.PyFuncModel:
    """
    Fetches a trained MLflow model from a remote server.

    Args:
        server_uri (str): The URI of the MLflow server to connect to.
        model_name (str): The name of the MLflow model to fetch.
        model_version (str): The version of the MLflow model to fetch.

    Returns:
        mlflow.pyfunc.PyFuncModel: The trained MLflow model.

    Raises:
        Exception: If the model cannot be fetched.

    """
    os.environ["MLFLOW_TRACKING_URI"] = server_uri
    endpoint = os.environ["AWS_S3_ENDPOINT"]
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"https://{endpoint}"
    try:
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )
        return model
    except Exception as error:
        raise Exception(
            f"Failed to fetch model {model_name} version {model_version}: {str(error)}"
        ) from error


def test_model(lib: str, server_uri: str, model_name: str, model_version: str) -> None:
    """
    Tests a trained MLflow model by making a prediction on sample data.

    Args:
        lib (str): The sample data to use for testing the model.
        server_uri (str): The URI of the MLflow server where the model is deployed.
        model_name (str): The name of the MLflow model to test.
        model_version (str): The version of the MLflow model to test.

    Returns:
        None

    Raises:
        Exception: If the model cannot be fetched or if the prediction fails.

    """
    # Step 1: Fetch the model from the MLflow server.
    model = fetch_model(server_uri, model_name, model_version)

    # Step 2: Make a prediction using the fetched model and the sample data.
    prediction = model.predict(lib)
    print(f"The prediction is: {prediction}")


def main() -> None:
    """
    The main entry point for the application.

    Returns:
        None

    Raises:
        Exception: If there is an error during model testing.

    """
    # Define the text to be used for testing the model.
    text = "lmnp AUTO_NaN NAT_SICORE_ NaN SURF_NaN EVT_SICORE_NaN"

    # Define the server URI where the model is deployed.
    server_uri = "https://projet-ape-4549.user.lab.sspcloud.fr"

    # Define the name and version of the model to be tested.
    model_name = "FastText-APE"
    version = "3"

    # Test the model with the provided text.
    test_model(text, server_uri, model_name, version)


if __name__ == "__main__":
    main()
