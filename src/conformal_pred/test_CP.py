import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.conformal_pred.conformal_pred import ConformalPredictorModule
from src.datasets import TextClassificationDataModule
from src.utils.mlflow import load_module_and_config


def run_test_CP(run_id, conformal_predictor_class):
    module, config = load_module_and_config(run_id)
    cfg: DictConfig = OmegaConf.create(config)
    trainer = hydra.utils.instantiate(cfg.model.trainer)

    datamodule = TextClassificationDataModule(
        cfg.data, cfg.tokenizer, cfg.dataset, batch_size=256, num_val_samples=10000
    )

    datamodule.setup()
    val_loader = datamodule.val_dataloader()
    predictions = trainer.predict(module, val_loader)
    predictions_tensor = torch.cat(predictions)

    true_values = torch.tensor(datamodule.df_val.apet_finale.values)

    conformal_predictor = conformal_predictor_class(
        module=module,
        val_labels=true_values.cpu(),
        val_predictions=predictions_tensor.cpu(),
        manually_coding_threshold=0.3,
        confidence_threshold=0.95,
    )

    conformal_predictor.compute_score_threshold()

    test_loader = datamodule.test_dataloader()
    conformer_model = ConformalPredictorModule(
        conformal_predictor=conformal_predictor,
    )
    results = trainer.predict(conformer_model, test_loader)

    avg_size_prediction_sets = [r["avg_size"] for r in results if r["avg_size"] is not None]
    avg_accuracy = [r["accuracy"] for r in results if r["accuracy"] is not None]

    print("Taille moyenne des ensembles de prédiction : ", np.mean(avg_size_prediction_sets))
    print("Taille médiane des ensembles de prédiction : ", np.median(avg_size_prediction_sets))
    print("Précision moyenne : ", np.mean(avg_accuracy))
    print("Précision médiane : ", np.median(avg_accuracy))
    # Plot the histogram of the sizes of the prediction sets
    plt.hist(avg_size_prediction_sets, bins=50)
    plt.xlabel("Taille des ensembles de prédiction")
    plt.ylabel("Fréquence")
    plt.title("Histogramme des tailles des ensembles de prédiction")
    plt.show()

    # Plot the histogram of the accuracies
    plt.hist(avg_accuracy, bins=50)
    plt.xlabel("Précision")
    plt.ylabel("Fréquence")
    plt.title("Histogramme des précisions")
    plt.show()
