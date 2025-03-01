import logging
import os

import hydra
import mlflow
import pandas as pd
import sklearn
from omegaconf import DictConfig, OmegaConf

from framework_classes import (
    EVALUATORS,
    PREPROCESSORS,
)
from utils.data import get_df_naf, get_processed_data, get_test_data, get_Y
from utils.mlflow import create_or_restore_experiment, log_dict
from utils.validation_viz import calibration_curve, confidence_histogram, sort_and_get_pred

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def evaluate(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    ##### Data #########

    # Sirene 4 test data - similar to the training data
    _, _, df_test = get_processed_data()  # already preprocessed

    # External test set - "gold standard"
    Y = get_Y(revision=cfg_dict["data"]["revision"])
    df_test_ls = get_test_data(**cfg_dict["data"], y=Y)
    df_naf = get_df_naf(revision=cfg_dict["data"]["revision"])

    preprocessor = PREPROCESSORS[cfg_dict["model"]["preprocessor"]]()

    df_test_ls = pd.concat(
        preprocessor.preprocess(
            df_test_ls,
            df_naf=df_naf,
            y=Y,
            text_feature=cfg_dict["data"]["text_feature"],
            textual_features=cfg_dict["data"]["textual_features"],
            categorical_features=cfg_dict["data"]["categorical_features"],
            test_size=0.1,
        ),
        axis=0,
    )

    ###### Fetch module from MLFlow ######

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    exp_name = cfg_dict["mlflow"]["experiment_name"]
    create_or_restore_experiment(exp_name)
    mlflow.set_experiment(exp_name)
    run_id = cfg_dict["model"]["test_params"]["run_id"]
    module = mlflow.pytorch.load_model(run_id)
    evaluator = EVALUATORS[cfg_dict["model"]["name"]](module)

    def launch_evaluation(df, suffix, n_bins):
        suffix = "_" + suffix
        mlflow.log_param("num_test_observation" + suffix, len(df))
        test_res = evaluator.launch_test(
            df,
            text_feature=cfg_dict["data"]["text_feature"],
            categorical_features=cfg_dict["data"]["categorical_features"],
            Y=Y,
            batch_size=cfg_dict["model"]["test_params"]["test_batch_size"],
            num_workers=os.cpu_count() - 1,
        )[0]
        for key, value in test_res.items():
            mlflow.log_metric(key + suffix, value)

        predictions, inference_time = evaluator.get_preds(
            df=df,
            Y=Y,
            return_inference_time=True,
            **cfg_dict["data"],
            batch_size=cfg_dict["model"]["test_params"]["test_batch_size"],
            num_workers=os.cpu_count() - 1,
        )
        mlflow.log_metric("inference_time" + suffix, inference_time)

        # Some results
        sorted_confidence, well_predicted, predicted_confidence, predicted_class, true_values = (
            sort_and_get_pred(predictions=predictions, df=df, Y=Y)
        )
        fig1 = confidence_histogram(sorted_confidence, well_predicted, df=df)
        fig2 = calibration_curve(
            n_bins=n_bins,
            confidences=predicted_confidence,
            predicted_classes=predicted_class,
            true_labels=true_values,
        )
        mlflow.log_figure(fig1, "confidence_histogram" + suffix + ".png")
        mlflow.log_figure(fig2, "calibration_curve" + suffix + ".png")

        brier_score = sklearn.metrics.brier_score_loss(
            well_predicted, predicted_confidence.numpy(), sample_weight=None, pos_label=1
        )
        mlflow.log_metric("brier_score" + suffix, brier_score)

        return predictions

    ##### Launch evaluation #####
    mlflow.set_experiment(exp_name + "_test")
    with mlflow.start_run():
        log_dict(cfg_dict)

        ### Label Studio test data ###
        logger.info("Launching evaluation on Label Studio test data.")
        predictions = launch_evaluation(df_test_ls, suffix="ls", n_bins=40)

        # Predictions in readable format - with true NACE code
        df_res = evaluator.get_aggregated_preds(df=df_test_ls, predictions=predictions, Y=Y)
        mlflow.log_table(df_res, "predictions.json")

        ### Sirene 4 test data ###
        logger.info("Launching evaluation on Sirene 4 test data.")
        predictions = launch_evaluation(df_test, suffix="s4", n_bins=100)

    return


if __name__ == "__main__":
    evaluate()
