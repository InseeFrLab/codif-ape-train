"""
Test stub.
"""

import mlflow
import pandas as pd

from base.evaluator import Evaluator
from base.preprocessor import Preprocessor


def run_test(test: dict, preprocessor: Preprocessor, evaluator: Evaluator) -> None:
    """
    Runs a test and logs the results using MLflow.

    Args:
        test (dict): A dictionary containing the test data and metadata. The
            dictionary should have the following keys: "Titre" (the test
            title), "Data" (a list of lists representing the test data),
            and "Expected_Output" (the expected output of the test).
        preprocessor (Preprocessor): A Preprocessor with a method called "clean_lib" that
            takes a pandas DataFrame as input and returns a cleaned version of
            it.
        evaluator (Evaluator): An Evaluator with a method called
            "get_aggregated_preds" that takes a pandas DataFrame as input and
            returns a pandas DataFrame with predictions and probabilities.

    Returns:
        None
    """
    # Convert the test data to a pandas DataFrame
    df = pd.DataFrame(test["Data"], index=[0])

    # Clean the data using the preprocessor
    df = preprocessor.clean_lib(df=df, text_feature=df.columns[0], method="evaluation")
    df.fillna(value="NaN", inplace=True)

    # Get predictions and probabilities using the evaluator
    preds = evaluator.get_aggregated_preds(
        df,
        y=df.columns[-1],
        text_feature=df.columns[0],
        categorical_features=df.columns[1:-1].to_list(),
        k=2,
    )

    # Compute the test result and information content
    res = int(preds[df.columns[-1]].iloc[0] == preds["predictions_5_k1"].iloc[0])
    score = preds["probabilities_k1"].iloc[0] - preds["probabilities_k2"].iloc[0]

    # Log the test result and information content using MLflow
    mlflow.log_metric(f"""{test["Titre"]} - Result""", res)
    mlflow.log_metric(f"""{test["Titre"]} - IC""", score)
