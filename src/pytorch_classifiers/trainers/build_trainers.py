import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)


def build_lightning_trainer(patience_early_stopping, num_epochs, **kwargs):
    # Trainer callbacks
    checkpoints = [
        {
            "monitor": "validation_loss_epoch",
            "save_top_k": 1,
            "save_last": False,
            "mode": "min",
        }
    ]
    callbacks = [ModelCheckpoint(**checkpoint) for checkpoint in checkpoints]
    callbacks.append(
        EarlyStopping(
            monitor="validation_loss_epoch",
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
        num_sanity_val_steps=2,
        strategy=strategy,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )
    return trainer


def build_transformers_trainer(patience_early_stopping, num_epochs):
    return
