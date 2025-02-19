import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.mlflow import MLFlowLogger


def build_lightning_trainer(
    patience_early_stopping, num_epochs, experiment_name, **kwargs
):
    # Trainer callbacks
    checkpoints = [
        {
            "monitor": "val_loss",
            "save_top_k": 1,
            "save_last": False,
            "mode": "min",
        }
    ]
    callbacks = [ModelCheckpoint(**checkpoint) for checkpoint in checkpoints]
    callbacks.append(
        EarlyStopping(
            monitor="val_loss",
            patience=patience_early_stopping,
            mode="min",
        )
    )
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # Strategy
    strategy = "auto"
    # Trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        max_epochs=num_epochs,
        num_sanity_val_steps=0,
        strategy=strategy,
        log_every_n_steps=1,
        enable_progress_bar=True,
        progress_bar_refresh_rate=20,
        profiler="simple",
        accelerator="gpu",
    )
    return trainer


def build_transformers_trainer(patience_early_stopping, num_epochs):
    return
