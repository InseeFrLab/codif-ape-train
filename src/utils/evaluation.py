import mlflow
import numpy as np
import pandas as pd
import sklearn
import torch

from .data import get_df_naf, mappings
from .validation_viz import (
    confidence_histogram,
    get_automatic_accuracy,
    plot_automatic_coding_accuracy_curve,
)


def run_evaluation(trainer, module, revision, Y, zipped_data):
    """
        Higher-level overload of the run_eval function, running the predictions beforehand.
        Main method of this file.

        Should be run within a mlflow.start_run() context (it is logging artifacts).

    Args:
        trainer (pl.Trainer): PyTorch Lightning Trainer.
        module (torchFastTextClassifier): TRAINED PyTorch Lightning Module.
        revision (str): Revision of the NAF code.
        Y (str): Column name of the target variable.
        ____
        zipped_data (iterable): Iterable of tuples containing the dataframes + dataloaders + suffixes (tags for logging).
            Ex: zipped_data = zip([df_val, df_test], [val_dataloader, test_dataloader], ["val", "test"])
        ____
    """
    df_naf = get_df_naf(revision=revision)
    # fasttext_preds_labels, fasttext_preds_scores = get_fasttext_preds(revision=revision)
    fasttext_preds_labels, fasttext_preds_scores = None, None
    for df, dataloader, suffix in zipped_data:
        predictions = trainer.predict(module, dataloader)  # accumulates predictions over batches
        predictions_tensor = torch.cat(predictions).cpu()  # (num_test_samples, num_classes)

        mlflow.log_metric("ece_" + suffix, module.ece.compute())
        fig, ax = module.ece.plot()
        mlflow.log_figure(
            fig,
            "calibration_curve_" + suffix + ".png",
        )

        run_eval(
            df=df,
            df_naf=df_naf,
            predictions_tensor=predictions_tensor,
            fasttext_preds_labels=fasttext_preds_labels,
            fasttext_preds_scores=fasttext_preds_scores,
            revision=revision,
            Y=Y,
            suffix=suffix,
        )


######### Utils for the run_evaluation method ##########


def run_eval(
    df,
    df_naf,
    predictions_tensor,
    fasttext_preds_labels,
    fasttext_preds_scores,
    revision,
    Y,
    suffix="val",
    n_bins=100,
):
    """
    Evaluation method independent from the model (requires to have computed the predictions beforehand).
    """
    true_values = get_ground_truth(df, Y)

    (
        sorted_confidence,
        predicted_confidence,
        predicted_class,
    ) = sort_and_get_pred(predictions=predictions_tensor, true_values=true_values)
    well_predicted = get_well_predicted_mask(predicted_class, true_values)

    fig1 = confidence_histogram(sorted_confidence, well_predicted, df=df)
    mlflow.log_figure(fig1, "confidence_histogram_" + suffix + ".png")

    brier_score = sklearn.metrics.brier_score_loss(
        well_predicted, predicted_confidence.numpy(), sample_weight=None, pos_label=1
    )
    mlflow.log_metric("brier_score" + suffix, brier_score)

    # Get and log accuracy at all levels
    naf_predictions = get_pred_nafs(predicted_class, revision=revision)
    all_level_preds = get_all_levels(naf_predictions, df_naf)
    all_level_ground_truth = get_all_levels(df, df_naf, col=Y, revision=revision)

    accs = (all_level_preds == all_level_ground_truth).mean(axis=0)

    for i in range(len(accs)):
        mlflow.log_metric(f"accuracy_{suffix}_level_{i+1}", accs.iloc[i])

    if suffix == "test":
        thresholds = np.linspace(0, 1, 100)
        torchft_scores = get_confidence_score(sorted_confidence)
        torchft_plot = get_automatic_accuracy(
            thresholds,
            torch.clamp(torchft_scores, 0, 1).numpy(),
            predicted_class.numpy(),
            true_values,
        )

        if fasttext_preds_labels is not None:
            ft_plot = get_automatic_accuracy(
                thresholds,
                np.clip(fasttext_preds_scores.values.reshape(-1), 0, 1),
                fasttext_preds_labels.values.reshape(-1),
                true_values,
            )
        else:
            ft_plot = None

        fig = plot_automatic_coding_accuracy_curve(torchft_plot, ft_plot, thresholds)
        mlflow.log_figure(fig, "automatic_coding_accuracy_curve.png")

    return


def get_confidence_score(sorted_confidence):
    return sorted_confidence[:, 0] - sorted_confidence[:, 1:5].sum(axis=1)


# def get_fasttext_preds(revision):
#     """
#     Get the fastText (production API) predictions for the given revision.

#     Args:
#         revision (str): The revision of the NAF code.
#             Must be either "NAF2008" or "NAF2025".
#     Returns:
#         fasttext_preds_labels (pd.DataFrame): DataFrame with the predicted labels, tokenized (int)
#         fasttext_preds_scores (pd.DataFrame): DataFrame with the predicted scores (float).


#     """

#     # As of 05/21/2025, NAF2025 API is broken so no fasttext preds / labels
#     if revision == "NAF2025":
#         return None, None

#     fs = get_file_system()

#     # take the path of "df_test.parquet", remove the extension...
#     df_res_ft = pd.read_parquet(PATHS[revision][-2][:-8] + "_predictions_ft.parquet", filesystem=fs)
#     mapping = get_label_mapping(revision)

#     fasttext_preds_labels = df_res_ft[["APE_NIV5_pred_k1"]]
#     fasttext_preds_scores = df_res_ft[["proba_k1"]]
#     fasttext_preds_labels_int = fasttext_preds_labels.map(mapping.get)

#     return fasttext_preds_labels_int, fasttext_preds_scores


def get_label_mapping(revision):
    """
    Get the label mapping for the given revision.

    Args:
        revision (str): The revision of the NAF code.
            Must be either "NAF2008" or "NAF2025".
    Returns:
        dict[str] -> int : The label mapping for the given revision.

    """
    assert revision in [
        "NAF2008",
        "NAF2025",
    ], f"Invalid revision: {revision} - must be NAF2008 or NAF2025."
    if revision == "NAF2008":
        key = "apet_finale"
    else:
        key = "nace2025"

    ape_niv5_mapping = mappings[key]

    return ape_niv5_mapping


def get_inv_mapping(revision):
    """
    Get the inverse label mapping for the given revision.
    Args:
        revision (str): The revision of the NAF code.
            Must be either "NAF2008" or "NAF2025".
    Returns:
        dict[int] -> str : The inverse label mapping for the given revision.

    """

    ape_niv5_mapping = get_label_mapping(revision)
    inv_ape_niv5_mapping = {v: k.upper() for k, v in ape_niv5_mapping.items()}

    return inv_ape_niv5_mapping


def get_ground_truth(df, Y):
    true_values = df[Y].values
    return true_values


def get_well_predicted_mask(predicted_class, true_values):
    well_predicted = (predicted_class == true_values).float()
    return well_predicted


def sort_and_get_pred(predictions, true_values):
    sorted_confidence, sorted_confidence_indices = predictions.sort(descending=True)

    predicted_class = sorted_confidence_indices[:, 0]
    predicted_confidence = sorted_confidence[:, 0]

    return sorted_confidence, predicted_confidence, predicted_class


def get_all_levels(df, df_naf, col="APE_NIV5_pred", revision=None):
    """
    Get all levels of the NAF code from the dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing the NAF codes.
        df_naf (pd.DataFrame): DataFrame containing the NAF data.

    Returns:
        pd.DataFrame: DataFrame with all levels of the NAF code.
    """
    df = df[[col]]

    if df[col].dtype == "int64":
        inv_ape_niv5_mapping = get_inv_mapping(revision=revision)
        df[col] = df[col].map(inv_ape_niv5_mapping)

    df_full = df.merge(df_naf, left_on=col, right_on="APE_NIV5").drop(columns=[col, "LIB_NIV5"])

    return df_full


def get_pred_nafs(predicted_class, revision):
    """
    Get the predicted NAF codes from the predicted class.

    Args:
        predicted_class (torch.Tensor): Predicted class tensor.
        revision (str): The revision of the NAF code.
            Must be either "NAF2008" or "NAF2025".

    Returns:
        pd.DataFrame: DataFrame with the predicted NAF codes.
    """
    inv_ape_niv5_mapping = get_inv_mapping(revision=revision)
    df_pred = pd.DataFrame(predicted_class.numpy(), columns=["APE_NIV5_pred"])
    df_pred["APE_NIV5_pred"] = df_pred["APE_NIV5_pred"].map(inv_ape_niv5_mapping)

    return df_pred
