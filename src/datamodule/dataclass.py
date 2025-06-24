from pathlib import Path
from typing import Dict, Literal, Optional

import torch
from clearml import Dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.configs.config import ExperimentConfig
from src.configs.constants import PROJECT_ROOT
from src.configs.logger import LOGGER
from src.datamodule.dataset import ClassificationDataset
from src.datamodule.transform import get_train_transform, get_val_transform


class ClassificationDataModule(LightningDataModule):

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self._train_transform = get_train_transform(
            *config.data_config.img_size,
            config=config,
        )
        self._val_transform = get_val_transform(*config.data_config.img_size)

        self.save_hyperparameters(logger=False)

        self.data_path: Path = PROJECT_ROOT / '.data_cache'
        self.initialized = False

        self.train_dataset: Optional[ClassificationDataset] = None
        self.val_dataset: Optional[ClassificationDataset] = None
        self.test_dataset: Optional[ClassificationDataset] = None

    @property
    def class_to_idx(self) -> Dict[str, int]:
        if not self.initialized:
            self.prepare_data()
            self.setup('test')
        if self.test_dataset is None:
            raise ValueError('Test dataset not initialized')
        return self.test_dataset.class_to_idx

    def prepare_data(self) -> None:
        dataset = Dataset.get(
            dataset_name=self.config.data_config.dataset_name,
        )
        local_path = dataset.get_local_copy()
        self.data_path = Path(local_path)
        LOGGER.info(f"Data path: {self.data_path}")

    def setup(self, stage: Literal['fit', 'test']) -> None:
        if stage == 'fit':
            all_data = ClassificationDataset(
                root=Path(self.data_path, 'train'),
                transform=self._train_transform,
            )
            train_split = int(
                len(all_data) * self.config.data_config.data_split[0],
            )
            val_split = len(all_data) - train_split
            splits = torch.utils.data.random_split(
                all_data,
                [train_split, val_split],
            )
            self.train_dataset = splits[0]
            self.val_dataset = splits[1]
            self.val_dataset = self._val_transform.transforms

        elif stage == 'test':
            self.test_dataset = ClassificationDataset(
                Path(self.data_path, 'test'),
                transform=self._val_transform,
            )

        self.initialized = True

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data_config.batch_size,
            num_workers=self.config.data_config.num_workers,
            pin_memory=self.config.data_config.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data_config.batch_size,
            num_workers=self.config.data_config.num_workers,
            pin_memory=self.config.data_config.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config.data_config.batch_size,
            num_workers=self.config.data_config.num_workers,
            pin_memory=self.config.data_config.pin_memory,
            shuffle=False,
        )
