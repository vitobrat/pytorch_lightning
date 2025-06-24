from pathlib import Path
from typing import Tuple, Union

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, model_validator


class _BaseValidatedConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')


class ModuleConfig(_BaseValidatedConfig):
    model_name: str
    pretrained: bool
    model_kwargs: dict[str, Union[int, float, str]]
    lr: float


class DataConfig(_BaseValidatedConfig):
    dataset_cache: bool
    img_size: Tuple[int, int]
    batch_size: int
    train_split: float
    num_workers: int
    pin_memory: bool
    dataset_name: str
    hue_shift_limit: int = 20
    sat_shift_limit: int = 30
    val_shift_limit: int = 20
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2

    @model_validator(mode='after')
    def splits_check_up_to_one(self) -> 'DataConfig':
        if not (0 < self.train_split <= 1):
            raise ValueError(
                f"Split must be between 0 and 1, got {self.train_split}",
            )

        return self


class TrainingConfig(_BaseValidatedConfig):
    min_epochs: int
    max_epochs: int
    val_every_n_epochs: int
    log_every_n_epochs: int
    deterministic: bool
    task: str = 'multiclass'
    average: str = 'macro'
    start_factor: float = 1.0
    end_factor: float = 0.01
    seed: int = 0


class ExperimentConfig(_BaseValidatedConfig):
    project_name: str
    experiment_name: str
    track_clearml: bool
    trainer_config: TrainingConfig
    data_config: DataConfig
    module_config: ModuleConfig

    @classmethod
    def from_yaml(cls, path: Path | str) -> 'ExperimentConfig':
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
