from pathlib import Path
from typing import Tuple

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, model_validator


class _BaseValidatedConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')


class ModuleConfig(_BaseValidatedConfig):
    model_name: str
    pretrained: bool
    model_kwargs: dict
    lr: float


class DataConfig(_BaseValidatedConfig):
    dataset_url: str
    dataset_cache: bool
    img_size: Tuple[int, int]
    batch_size: int
    data_split: Tuple[float, ...]
    num_workers: int
    pin_memory: bool

    @model_validator(mode='after')
    def splits_sum_up_to_one(self):
        epsilon = 1e6
        total = sum(self.data_split)
        if abs(total - 1.0) > epsilon:
            raise ValueError(f"Splits should sum up to one, got {total}")

        return self


class TrainingConfig(_BaseValidatedConfig):
    min_epochs: int
    max_epochs: int
    val_every_n_epochs: int
    log_every_n_epochs: int
    deterministic: bool


class ExperimentConfig(_BaseValidatedConfig):
    project_name: str
    experiment_name: str
    track_clearml: bool
    trainer_config: TrainingConfig
    data_config: DataConfig
    module_config: ModuleConfig

    @classmethod
    def from_yaml(cls, path: Path) -> 'ExperimentConfig':
        config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**config)

    def to_yaml(self, path: Path) -> None:
        with open(path, 'w') as output_file:
            yaml.safe_dump(
                self.model_dump(),
                output_file,
                default_flow_style=False,
                sort_keys=False,
            )
