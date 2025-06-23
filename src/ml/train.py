import os
from pathlib import Path

import lightning
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from src.configs.config import ExperimentConfig
from src.configs.constants import PROJECT_ROOT
from src.datamodule.dataclass import ClassificationDataModule
from src.ml.callback.experiment_tracking import ClearMLTracking
from src.ml.lightning_module import ClassificationLightningModule


def train(config: ExperimentConfig) -> None:
    lightning.seed_everything(config.trainer_config.seed)
    datamodule = ClassificationDataModule(config)

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            save_top_k=3,
            monitor='valid_f1',
            mode='max',
            every_n_epochs=1,
        ),
    ]
    if config.track_clearml:
        tracking_callback = ClearMLTracking(config, datamodule.class_to_idx)
        callbacks += [tracking_callback]
    model = ClassificationLightningModule(config, datamodule.class_to_idx)

    trainer = Trainer(
        **dict(config.trainer_config),
        callbacks=callbacks,
        overfit_batches=60,
    )
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == '__main__':
    config_path = os.getenv(
        'TRAIN_CONFIG_PATH',
        Path(PROJECT_ROOT, 'configs', 'training.yaml'),
    )
    train(config=ExperimentConfig.from_yaml(config_path))
