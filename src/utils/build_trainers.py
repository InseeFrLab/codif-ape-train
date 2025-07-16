import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)


class CustomProgressBar(TQDMProgressBar):
    def __init__(self, refresh_rate=20):  # Set refresh rate to 10 batches
        super().__init__(refresh_rate=refresh_rate)


def build_lightning_trainer(patience_early_stopping, num_epochs, **kwargs):
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
    callbacks.append(CustomProgressBar())

    # Strategy
    strategy = "auto"
    # Trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        max_epochs=num_epochs,
        num_sanity_val_steps=0,
        strategy=strategy,
        log_every_n_steps=5,
        enable_progress_bar=True,
    )
    return trainer


def build_transformers_trainer(patience_early_stopping, num_epochs):
    return
