from typing import Any, Dict, List

import torch
import torch.nn.functional as func
from lightning import LightningModule
from timm import create_model
from torch import Tensor
from torchmetrics import MeanMetric

from src.configs.config import ExperimentConfig
from src.utils_module import get_metrics


class ClassificationLightningModule(LightningModule):
    def __init__(
        self,
        configs: ExperimentConfig,
        class_to_idx: Dict[str, int],
    ):
        super().__init__()
        self._configs = configs
        self._train_loss = MeanMetric()
        self._val_loss = MeanMetric()

        metrics = get_metrics(
            num_classes=len(class_to_idx.keys()),
            num_labels=len(class_to_idx.keys()),
            task=configs.trainer_config.task,
            average=configs.trainer_config.average,
        )
        self._val_metrics = metrics.clone(prefix='val_')
        self._test_metrics = metrics.clone(prefix='test_')

        self.model = create_model(
            num_classes=len(class_to_idx.keys()),
            model_name=configs.module_config.model_name,
            pretrained=configs.module_config.pretrained,
            **configs.module_config.kwargs,
        )

        self.save_hyperparameters()

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def training_step(self, batch: List[Tensor]) -> Dict[str, Tensor]:
        images, labels = batch
        logging = self(images)
        loss = func.cross_entropy(logging, labels)
        self._train_loss(loss)
        self.log(
            'step_loss',
            self._train_loss.compute(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return {'loss': loss}

    def on_train_epoch_end(self) -> None:
        self.log(
            'mean_train_loss',
            self._train_loss.compute(),
            on_step=False,
            prog_bar=True,
            on_epoch=True,
        )
        self._train_loss.reset()

    def validation_step(self, batch: List[Tensor]) -> None:
        images, labels = batch
        logits = self(images)
        loss = func.cross_entropy(logits, labels)
        self._val_loss(loss)

        self._val_metrics(logits, labels)

    def on_validation_epoch_end(self) -> None:
        self.log(
            'mean_val_loss',
            self._val_loss.compute(),
            on_step=False,
            prog_bar=True,
            on_epoch=True,
        )

        self.log_dict(
            self._val_metrics.compute(),
            prog_bar=True,
            on_epoch=True,
        )
        self._val_metrics.reset()

    def test_step(self, batch: List[Tensor]) -> Tensor:
        images, labels = batch
        logits = self(images)

        preds = torch.argmax(logits, dim=1)
        self._test_metrics(logits, labels)
        return preds

    def on_test_epoch_end(self) -> None:
        self.log(self._test_metrics.compute(), prog_bar=True, on_epoch=True)
        self._test_metrics.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self._configs.trainer_config.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=self._configs.trainer_config.start_factor,
            end_factor=self._configs.trainer_config.end_factor,
            total_iters=self._configs.trainer_config.max_epochs,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            },
        }
