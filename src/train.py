"""
Main script.
"""
import os
import time
import argparse
import yaml
import mlflow
import pyarrow.parquet as pq

from constants import FRAMEWORK_CLASSES, TEXT_FEATURE
from fasttext_classifier.fasttext_wrapper import FastTextWrapper
from tests.test_main import run_test

from utils.data import get_file_system

parser = argparse.ArgumentParser(
    description="FastAPE 🚀 : Model for coding a company's main activity"
)
parser.add_argument(
    "--remote_server_uri",
    type=str,
    default="https://projet-ape-mlflow.user.lab.sspcloud.fr",
    help="MLflow URI",
    required=True,
)
parser.add_argument(
    "--experiment_name",
    type=str,
    choices=["experimentation", "Production", "test"],
    default="test",
    help="Experiment name in MLflow",
)
parser.add_argument(
    "--run_name",
    type=str,
    default="default",
    help="Run name in MLflow",
)
parser.add_argument(
    "--Y",
    type=str,
    choices=["APE_NIV5", "APE_NIV4", "APE_NIV3", "APE_NIV2", "APE_NIV1", "apet_finale"],
    default="APE_NIV5",
    help="Target name",
    required=True,
)
parser.add_argument(
    "--dim",
    type=int,
    default=180,
    help="Size of word vectors",
    required=True,
)
parser.add_argument(
    "--lr", type=float, default=0.2, metavar="LR", help="Learning rate (default: 0.2)"
)
parser.add_argument(
    "--epoch", type=int, default=50, metavar="N", help="Number of epochs to train (default: 50)"
)
parser.add_argument(
    "--wordNgrams", type=int, default=3, metavar="N", help="Max length of word ngram (default: 3)"
)
parser.add_argument(
    "--minn", type=int, default=3, metavar="N", help="Min length of char ngram (default: 3)"
)
parser.add_argument(
    "--maxn", type=int, default=3, metavar="N", help="Max length of char ngram (default: 4)"
)
parser.add_argument(
    "--minCount",
    type=int,
    default=3,
    metavar="N",
    help="Minimal number of word occurrences (default: 3)",
)
parser.add_argument(
    "--bucket", type=int, default=3, metavar="N", help="Number of buckets (default: 2000000)"
)
parser.add_argument(
    "--loss",
    type=str,
    choices=["ns", "hs", "softmax", "ova"],
    default="ova",
    help="Loss function",
    required=True,
)
parser.add_argument(
    "--label_prefix",
    type=str,
    default="__label__",
    help="Labels prefix",
    required=True,
)
parser.add_argument(
    "--categorical_features_1",
    type=str,
    default="AUTO",
    help="Type of observation",
    required=True,
)
parser.add_argument(
    "--categorical_features_2",
    type=str,
    default="NAT_SICORE",
    help="Nature of observation",
    required=True,
)
parser.add_argument(
    "--categorical_features_3",
    type=str,
    default="SURF",
    help="Surface of the company",
    required=True,
)
parser.add_argument(
    "--categorical_features_4",
    type=str,
    default="EVT_SICORE",
    help="Event of the observation",
    required=True,
)
args = parser.parse_args()


def main(
    remote_server_uri: str,
    experiment_name: str,
    run_name: str,
    Y: str,
    dim: int,
    lr: float,
    epoch: int,
    wordNgrams: int,
    minn: int,
    maxn: int,
    minCount: int,
    bucket: int,
    loss: str,
    label_prefix: str,
    categorical_features_1: str,
    categorical_features_2: str,
    categorical_features_3: str,
    categorical_features_4: str,
    model_type: str = "fasttext",
):
    """
    Main method.
    """
    params = {
        key: value
        for key, value in locals().items()
        if (key not in ["remote_server_uri", "experiment_name", "run_name", "Y", "model_type"])
        and not key.startswith("categorical_features")
    }
    params["thread"] = os.cpu_count()
    categorical_features = [
        value for key, value in locals().items() if key.startswith("categorical_features")
    ]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)
    fs = get_file_system()

    with mlflow.start_run(run_name=run_name):
        framework_classes = FRAMEWORK_CLASSES[model_type]

        preprocessor = framework_classes["preprocessor"]()
        trainer = framework_classes["trainer"]()

        print("\n\n*** 1- Preprocessing the database...\n")
        t = time.time()
        # Load data
        df = pq.read_table(
            "projet-ape/extractions/20240124_sirene4.parquet", filesystem=fs
        ).to_pandas()

        # Preprocess data
        df_train, df_test = preprocessor.preprocess(
            df=df,
            y=Y,
            text_feature=TEXT_FEATURE,
            categorical_features=categorical_features,
        )
        print(f"*** Done! Preprocessing lasted {round((time.time() - t)/60,1)} minutes.\n")

        # Run training of the model
        print("*** 2- Training the model...\n")
        t = time.time()
        model = trainer.train(df_train, Y, TEXT_FEATURE, categorical_features, params)
        print(f"*** Done! Training lasted {round((time.time() - t)/60,1)} minutes.\n")

        if model_type == "fasttext":
            fasttext_model_path = run_name + ".bin"
            model.save_model(fasttext_model_path)

            artifacts = {
                "fasttext_model_path": fasttext_model_path,
                "train_data": "train_text.txt",
            }

            mlflow.pyfunc.log_model(
                artifact_path=run_name,
                code_path=["src/fasttext_classifier/", "src/base/"],
                python_model=FastTextWrapper(),
                artifacts=artifacts,
            )
        elif model_type == "pytorch":
            mlflow.pytorch.log_model(pytorch_model=model, artifact_path=run_name)
        else:
            raise KeyError("Model type is not valid.")

        # Log parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_param("features", categorical_features)
        mlflow.log_param("Y", Y)

        # Evaluation
        print("*** 3- Evaluating the model...\n")
        t = time.time()
        if model_type == "fasttext":
            evaluator = framework_classes["evaluator"](model)
        elif model_type == "pytorch":
            evaluator = framework_classes["evaluator"](model, trainer.tokenizer)
        else:
            raise KeyError("Model type is not valid.")

        accuracies = evaluator.evaluate(df_test, Y, TEXT_FEATURE, categorical_features, 5)

        # Log metrics
        for metric, value in accuracies.items():
            mlflow.log_metric(metric, value)

        print(f"*** Done! Evaluation lasted {round((time.time() - t)/60,1)} minutes.\n")

        # Tests
        print("*** 4- Performing standard tests...\n")
        t = time.time()
        with open("src/tests/tests.yaml", "r", encoding="utf-8") as stream:
            tests = yaml.safe_load(stream)
        for case in tests.keys():
            run_test(tests[case], preprocessor, evaluator)

        print(f"*** Done! Tests lasted {round((time.time() - t)/60,1)} minutes.\n")


if __name__ == "__main__":
    main(**vars(args))
