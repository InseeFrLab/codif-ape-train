"""
Main script.
"""
import os
import time
import argparse
import yaml
import mlflow
import pandas as pd

from constants import FRAMEWORK_CLASSES
from utils.mappings import mappings
from camembert.custom_pipeline import CustomPipeline
from tests.test_main import run_test
from utils.data import get_sirene_4_data, get_test_data, get_sirene_3_data


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
    choices=["Experimentation", "Production", "Test"],
    default="Test",
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
    "--text_features_1",
    type=str,
    help="Description of company's activity",
    required=True,
)
parser.add_argument(
    "--text_features_2",
    type=str,
    default="activ_nat_lib_et",
    help="Other nature of observation description",
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
parser.add_argument(
    "--categorical_features_5",
    type=str,
    default="cj",
    help="CJ of the company",
    required=True,
)
parser.add_argument(
    "--categorical_features_6",
    type=str,
    default="activ_perm_et",
    help="Permanent or seasonal character of the activities of the company",
    required=True,
)
parser.add_argument(
    "--embedding_dim_1",
    type=int,
    default=3,
    help="Embedding dimension for type",
    required=True,
)
parser.add_argument(
    "--embedding_dim_2",
    type=int,
    default=3,
    help="Embedding dimension for nature",
    required=True,
)
parser.add_argument(
    "--embedding_dim_3",
    type=int,
    default=1,
    help="Embedding dimension for surface",
    required=True,
)
parser.add_argument(
    "--embedding_dim_4",
    type=int,
    default=3,
    help="Embedding dimension for event",
    required=True,
)
parser.add_argument(
    "--embedding_dim_5",
    type=int,
    default=3,
    help="Embedding dimension for cj",
    required=True,
)
parser.add_argument(
    "--embedding_dim_6",
    type=int,
    default=1,
    help="Embedding dimension for permanence",
    required=True,
)
parser.add_argument(
    "--pre_training_weights",
    type=str,
    default="camembert/camembert-base",
    help="Pre-training weights on Huggingface",
    required=True,
)
parser.add_argument(
    "--model_class",
    type=str,
    choices=[
        "fasttext",
        "pytorch",
        "camembert",
        "camembert_one_hot",
        "camembert_embedded",
    ],
    default="fasttext",
    help="Model type",
    required=True,
)
parser.add_argument(
    "--start_month",
    type=int,
    default=1,
    help="Start month for Sirene 3 data",
    required=True,
)
parser.add_argument(
    "--start_year",
    type=int,
    default=2018,
    help="Start year for Sirene 3 data",
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
    text_features_1: str,
    text_features_2: str,
    categorical_features_1: str,
    categorical_features_2: str,
    categorical_features_3: str,
    categorical_features_4: str,
    categorical_features_5: str,
    categorical_features_6: str,
    embedding_dim_1: int,
    embedding_dim_2: int,
    embedding_dim_3: int,
    embedding_dim_4: int,
    embedding_dim_5: int,
    embedding_dim_6: int,
    model_class: str,
    pre_training_weights: str,
    start_month: int,
    start_year: int,
):
    """
    Main method.
    """
    params = {
        key: value
        for key, value in locals().items()
        if (
            key
            not in [
                "remote_server_uri",
                "experiment_name",
                "run_name",
                "Y",
                "model_class",
                "text_features_1",
                "text_features_2",
                "pre_training_weights",
                "start_month",
                "start_year",
            ]
        )
        and not key.startswith("categorical_features")
        and not key.startswith("embedding_dim")
    }
    params["thread"] = os.cpu_count()
    categorical_features = [
        value for key, value in locals().items() if key.startswith("categorical_features")
    ]
    embedding_dims = [value for key, value in locals().items() if key.startswith("embedding_dim")]

    # Concatenate text features
    text_feature = f"{text_features_1} {text_features_2}"

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        framework_classes = FRAMEWORK_CLASSES[model_class]

        preprocessor = framework_classes["preprocessor"]()

        # Create trainer
        if model_class in ["camembert", "camembert_one_hot", "camembert_embedded"]:
            trainer = framework_classes["trainer"](pre_training_weights)
        else:
            trainer = framework_classes["trainer"]()

        print("\n\n*** 1- Preprocessing the database...\n")
        t = time.time()
        # Load data
        # Sirene 4
        df_s4 = get_sirene_4_data()
        # Sirene 3
        df_s3 = get_sirene_3_data(start_month=start_month, start_year=start_year)

        # Preprocess data
        # Sirene 4
        df_train_s4, df_test = preprocessor.preprocess(
            df=df_s4,
            y=Y,
            text_feature=text_feature,
            categorical_features=categorical_features,
            test_size=0.1,
        )
        # Get test_data from LabelStudio
        df_test_ls = pd.concat(
            preprocessor.preprocess(
                get_test_data(), Y, text_feature, categorical_features, add_codes=False
            ),
            axis=0,
        )
        # Sirene 3
        if df_s3.empty:
            df_train = df_train_s4
        else:
            df_train_s3 = pd.concat(
                preprocessor.preprocess(df_s3, Y, text_feature, categorical_features, recase=True),
                axis=0,
            )
            # All train data together
            df_train = pd.concat([df_train_s3, df_train_s4], axis=0).reset_index(drop=True)
        # Shuffle in case it is not done later
        df_train = df_train.sample(frac=1)
        mlflow.log_param("number_of_observations", df_train.shape[0])
        print(f"Number of observations in the training_set: {df_train.shape[0]}")
        print(f"*** Done! Preprocessing lasted {round((time.time() - t)/60,1)} minutes.\n")

        # Run training of the model
        print("*** 2- Training the model...\n")
        t = time.time()

        if model_class in ["camembert", "camembert_one_hot", "camembert_embedded"]:
            model = trainer.train(
                df_train,
                Y,
                text_feature,
                categorical_features,
                params,
                embedding_dims=embedding_dims,
            )
        else:
            model = trainer.train(df_train, Y, text_feature, categorical_features, params)
        print(f"*** Done! Training lasted {round((time.time() - t)/60,1)} minutes.\n")

        inference_params = {
            "k": 1,
        }
        # Infer the signature including parameters
        signature = mlflow.models.infer_signature(
            params=inference_params,
        )
        if model_class == "fasttext":
            fasttext_model_path = run_name + ".bin"
            model.save_model(fasttext_model_path)

            artifacts = {
                "fasttext_model_path": fasttext_model_path,
                "train_data": "train_text.txt",
            }

            mlflow.pyfunc.log_model(
                artifact_path=run_name,
                code_path=["src/fasttext_classifier/", "src/base/", "src/utils/"],
                python_model=framework_classes["wrapper"](text_feature, categorical_features),
                artifacts=artifacts,
                signature=signature,
            )
        elif model_class == "pytorch":
            mlflow.pytorch.log_model(pytorch_model=model, artifact_path=run_name)
        elif model_class in ["camembert", "camembert_one_hot", "camembert_embedded"]:
            model_output_dir = "./camembert_model"
            pipeline_output_dir = "./camembert_pipeline"
            model.save_model(model_output_dir)

            pipe = CustomPipeline(
                framework="pt",
                model=framework_classes["model"].from_pretrained(
                    model_output_dir,
                    num_labels=len(mappings.get("APE_NIV5")),
                    categorical_features=categorical_features,
                ),
                tokenizer=trainer.tokenizer,
            )

            pipe.save_pretrained(pipeline_output_dir)
            mlflow.pyfunc.log_model(
                artifacts={"pipeline": pipeline_output_dir},
                code_path=["src/camembert/", "src/base/", "src/utils/"],
                artifact_path=run_name,
                python_model=framework_classes["wrapper"](text_feature, categorical_features),
                signature=signature,
            )
        else:
            raise KeyError("Model type is not valid.")

        # Log additional params
        mlflow.log_param("features", categorical_features)
        mlflow.log_param("Y", Y)

        # Evaluation
        print("*** 3- Evaluating the model...\n")
        t = time.time()
        if model_class in ["fasttext", "camembert", "camembert_one_hot", "camembert_embedded"]:
            evaluator = framework_classes["evaluator"](model)
        elif model_class == "pytorch":
            evaluator = framework_classes["evaluator"](model, trainer.tokenizer)
        else:
            raise KeyError("Model type is not valid.")

        accuracies = evaluator.evaluate(df_test, Y, text_feature, categorical_features, 5)

        # Log metrics
        for metric, value in accuracies.items():
            mlflow.log_metric(metric, value)

        accuracies = evaluator.evaluate(df_test_ls, Y, text_feature, categorical_features, 5)

        # Log additional metrics
        for metric, value in accuracies.items():
            metric = "ls_" + metric
            mlflow.log_metric(metric, value)

        print(f"*** Done! Evaluation lasted {round((time.time() - t)/60,1)} minutes.\n")

        # Tests: dependent on categorical features
        if "cj" not in categorical_features:
            print("*** 4- Performing standard tests...\n")
            t = time.time()
            with open("src/tests/tests.yaml", "r", encoding="utf-8") as stream:
                tests = yaml.safe_load(stream)
            for case in tests.keys():
                run_test(tests[case], preprocessor, evaluator)

            print(f"*** Done! Tests lasted {round((time.time() - t)/60,1)} minutes.\n")


if __name__ == "__main__":
    main(**vars(args))
