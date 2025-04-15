import logging
import os
import sys

import hydra
import mlflow
import numpy as np
import pandas as pd
import sklearn
import torch
from omegaconf import DictConfig, OmegaConf
from torchFastText.datasets import FastTextModelDataset

from evaluators import Evaluator
from framework_classes import (
    DATASETS,
    LOSSES,
    MODELS,
    MODULES,
    OPTIMIZERS,
    SCHEDULERS,
    TOKENIZERS,
    TRAINERS,
)
from utils.data import PATHS, get_file_system, get_processed_data, get_Y
from utils.mappings import mappings
from utils.mlflow import create_or_restore_experiment, log_dict
from utils.validation_viz import (
    calibration_curve,
    confidence_histogram,
    get_automatic_accuracy,
    plot_automatic_coding_accuracy_curve,
    sort_and_get_pred,
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    create_or_restore_experiment(cfg_dict["mlflow"]["experiment_name"])
    mlflow.set_experiment(cfg_dict["mlflow"]["experiment_name"])

    with mlflow.start_run():
        # Log config
        log_dict(cfg_dict)

        ##### Data #########
        Y = get_Y(revision=cfg_dict["data"]["revision"])
        df_train, df_val, df_test = get_processed_data(revision=cfg_dict["data"]["revision"])

        df_train = df_train.sample(frac=0.001)
        df_val = df_val.sample(frac=0.01)
        df_test = df_test.sample(frac=0.01)

        mlflow.log_param("number_of_training_observations", df_train.shape[0])

        train_text, train_categorical_variables = (
            df_train[cfg_dict["data"]["text_feature"]].values,
            df_train[cfg_dict["data"]["categorical_features"]].values,
        )
        val_text, val_categorical_variables = (
            df_val[cfg_dict["data"]["text_feature"]].values,
            df_val[cfg_dict["data"]["categorical_features"]].values,
        )
        test_text, test_categorical_variables = (
            df_test[cfg_dict["data"]["text_feature"]].values,
            df_test[cfg_dict["data"]["categorical_features"]].values,
        )

        ###### Tokenizer ######

        tokenizer = TOKENIZERS[cfg_dict["tokenizer"]["tokenizer_name"]](
            **cfg_dict["tokenizer"], training_text=train_text
        )
        logger.info(tokenizer)

        ###### Dataset ######

        if cfg_dict["dataset"] is not None:
            similarity_coefficients = cfg_dict["dataset"].get("similarity_coefficients", None)

            dataset_class = DATASETS[cfg_dict["dataset"]["dataset_name"]]

            train_dataset = dataset_class(
                texts=train_text,
                categorical_variables=train_categorical_variables,
                tokenizer=tokenizer,
                outputs=df_train[Y].values,
                revision=cfg_dict["data"]["revision"],
                similarity_coefficients=similarity_coefficients,
            )
            val_dataset = dataset_class(
                texts=val_text,
                categorical_variables=val_categorical_variables,
                tokenizer=tokenizer,
                outputs=df_val[Y].values,
                revision=cfg_dict["data"]["revision"],
                similarity_coefficients=similarity_coefficients,
            )

            test_dataset = dataset_class(
                texts=test_text,
                categorical_variables=test_categorical_variables,
                tokenizer=tokenizer,
                outputs=df_test[Y].values,
                revision=cfg_dict["data"]["revision"],
                similarity_coefficients=similarity_coefficients,
            )

            if isinstance(train_dataset, FastTextModelDataset):
                train_dataloader = train_dataset.create_dataloader(
                    **cfg_dict["model"]["train_params"]
                )
                val_dataloader = val_dataset.create_dataloader(**cfg_dict["model"]["train_params"])
                test_dataloader = test_dataset.create_dataloader(
                    **cfg_dict["model"]["train_params"]
                )

        ###### Model #####
        num_classes = max(mappings[Y].values()) + 1

        categorical_vocab_sizes = []
        for feature in cfg_dict["data"]["categorical_features"]:
            if feature == "SRF":
                categorical_vocab_sizes.append(5)
            else:
                categorical_vocab_sizes.append(max(mappings[feature].values()) + 1)

        logger.info("Number of classes: " + str(num_classes))
        logger.info("categorical_vocab_sizes " + str(categorical_vocab_sizes))

        if cfg_dict["model"]["model_name"] == "torchFastText":
            # for torchFastText only, we add the number of words in the vocabulary
            # In general, tokenizer.num_tokens == num_rows is a invariant
            num_rows = tokenizer.num_tokens + tokenizer.get_nwords() + 1
            padding_idx = num_rows - 1

        # PyTorch model
        model = MODELS[cfg_dict["model"]["model_name"]](
            **cfg_dict["model"]["model_params"],
            tokenizer=tokenizer,
            num_rows=num_rows,
            num_classes=num_classes,
            categorical_vocabulary_sizes=categorical_vocab_sizes,
            padding_idx=padding_idx,
        )

        model = model.to(device)
        logger.info(model)

        # Lightning
        loss = LOSSES[cfg_dict["model"]["train_params"]["loss_name"]]().to(device)
        optimizer = OPTIMIZERS[
            cfg_dict["model"]["train_params"]["optimizer_name"]
        ]  # without the () !
        scheduler = SCHEDULERS[cfg_dict["model"]["train_params"]["scheduler_name"]]

        module = MODULES[cfg_dict["model"]["model_name"]](
            model=model,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            **cfg_dict["model"]["train_params"],
        )
        logger.info(module)

        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param("num_trainable_parameters", num_trainable)

        ###### Trainer #####
        trainer = TRAINERS[cfg_dict["model"]["train_params"]["trainer_name"]](
            **cfg_dict["model"]["train_params"],
        )

        if cfg_dict["model"]["preprocessor"] == "PyTorch":
            mlflow.pytorch.autolog()
            torch.cuda.empty_cache()
            torch.set_float32_matmul_precision("medium")

        trainer.fit(module, train_dataloader, val_dataloader)

        ########## Evaluation ##########
        logger.info("Starting evaluation...")
        fs = get_file_system()
        df_res_ft = pd.read_parquet(
            PATHS[cfg_dict["data"]["revision"]][-1][:-8] + "_predictions_ft.parquet", filesystem=fs
        )
        fasttext_preds_labels = df_res_ft["APE_NIV5_pred_k1"].values
        fasttext_preds_scores = df_res_ft["proba_k1"].values
        ground_truth = df_res_ft["APE_NIV5"]

        def run_eval(df, dataloader, suffix="val", n_bins=100):
            """
            Run evaluation on the given dataloader and log the results.
            """

            predictions = trainer.predict(
                module, dataloader
            )  # accumulates predictions over batches
            predictions_tensor = torch.cat(predictions).cpu()  # (num_test_samples, num_classes)

            (
                sorted_confidence,
                well_predicted,
                predicted_confidence,
                predicted_class,
                true_values,
            ) = sort_and_get_pred(predictions=predictions_tensor, df=df, Y=Y)
            fig1 = confidence_histogram(sorted_confidence, well_predicted, df=df)
            fig2 = calibration_curve(
                n_bins=n_bins,
                confidences=predicted_confidence,
                predicted_classes=predicted_class,
                true_labels=true_values,
            )
            mlflow.log_figure(fig1, "confidence_histogram_" + suffix + ".png")
            mlflow.log_figure(fig2, "calibration_curve_" + suffix + ".png")

            brier_score = sklearn.metrics.brier_score_loss(
                well_predicted, predicted_confidence.numpy(), sample_weight=None, pos_label=1
            )
            mlflow.log_metric("brier_score" + suffix, brier_score)

            # Use your aggregation function
            aggregated_results = Evaluator.get_aggregated_preds(
                df=df, Y=Y, predictions=predictions_tensor.numpy(), top_k=1
            )

            accuracy = Evaluator.compute_accuracies(
                aggregated_preds=aggregated_results, suffix=suffix
            )
            mlflow.log_metrics(accuracy)

            if suffix == "test":
                thresholds = np.linspace(0, 1, 100)
                torchft_scores = sorted_confidence[:, 0] - sorted_confidence[:, 1:5].sum(axis=1)
                torchft_plot = get_automatic_accuracy(
                    thresholds,
                    torch.clamp(torchft_scores, 0, 1).numpy(),
                    predicted_class.numpy(),
                    true_values,
                )
                ft_plot = get_automatic_accuracy(
                    thresholds,
                    np.clip(fasttext_preds_scores.reshape(-1), 0, 1),
                    fasttext_preds_labels.reshape(-1),
                    ground_truth.values,
                )

                fig = plot_automatic_coding_accuracy_curve(torchft_plot, ft_plot, thresholds)
                mlflow.log_figure(fig, "automatic_coding_accuracy_curve.png")

            return

        run_eval(df_val, val_dataloader, suffix="val")
        run_eval(df_test, test_dataloader, suffix="test")


if __name__ == "__main__":
    logger.info("GPU available: " + str(torch.cuda.is_available()))
    for i in range(len(sys.argv)):
        if sys.argv[-1] == "":  # Hydra may get an empty string
            logger.info("Removing empty string argument")
            sys.argv = sys.argv[:-1]  # Remove it
        else:
            break

    # Merge all the args into one
    args = " ".join(sys.argv)
    train()
