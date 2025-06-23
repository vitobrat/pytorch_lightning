import os
from typing import Dict, Optional

from clearml import OutputModel, Task
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from src.configs.config import ExperimentConfig
from src.configs.logger import LOGGER


class ClearMLTracking(Callback):
    def __init__(
        self,
        config: ExperimentConfig,
        class_to_idx: Optional[Dict[str, int]],
    ) -> None:
        super().__init__()
        self.config = config
        self.class_to_idx = class_to_idx
        self.task: Optional[Task] = None
        self.output_model: Optional[OutputModel] = None

    def on_fit_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self._setup_task()

    def on_test_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        final_checkpoint = select_checkpoint_for_export(trainer)
        LOGGER.info(f"Uploading checkpoint {final_checkpoint} to clearml")
        extension = final_checkpoint.split('.')[-1]

        if self.output_model is None:
            raise ValueError('Output model not initialized.')
        self.output_model.update_weights(
            weights_filename=final_checkpoint,
            target_filename=f"best.{extension}",
            auto_delete_file=True,
        )

    def _setup_task(self) -> None:
        Task.force_requirements_env_freeze()
        self.task = Task.init(
            project_name=self.config.project_name,
            task_name=self.config.experiment_name,
            output_uri=True,
            reuse_last_task_id=False,
            auto_connect_frameworks={'pytorch': False},
        )
        self.task.connect_configuration(configuration=self.config.model_dump())
        self.output_model = OutputModel(
            task=self.task,
            label_enumeration=self.class_to_idx,
        )


def select_checkpoint_for_export(trainer: Trainer) -> str:
    checkpoint_callback: Optional[ModelCheckpoint] = (
        trainer.checkpoint_callback
    )
    if checkpoint_callback is not None:
        checkpoint_path = checkpoint_callback.best_model_path
        if os.path.exists(checkpoint_path):
            LOGGER.info(f"Selected best checkpoint: {checkpoint_path}")
            return checkpoint_path
        else:
            LOGGER.warning(
                "Couldn't find the best checkpoint, "
                "probably callback haven't been called yet",
            )

    checkpoint_path = os.path.join(
        trainer.log_dir,
        'checkpoint-from-trainer.pth',
    )
    trainer.save_checkpoint(checkpoint_path)
    trainer.save_checkpoint(checkpoint_path)
    LOGGER.info(f"Saved checkpoint: {checkpoint_path}")
    return checkpoint_path
