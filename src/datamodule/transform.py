import albumentations as albu
from albumentations.pytorch import ToTensorV2

from src.configs.config import ExperimentConfig


def get_train_transform(
    image_width: int,
    image_height: int,
    config: ExperimentConfig,
) -> albu.Compose:
    return albu.Compose(
        [
            albu.Resize(height=image_height, width=image_width),
            albu.HorizontalFlip(p=0.5),
            albu.HueSaturationValue(
                hue_shift_limit=config.data_config.hue_shift_limit,
                sat_shift_limit=config.data_config.sat_shift_limit,
                val_shift_limit=config.data_config.val_shift_limit,
                p=0.5,
            ),
            albu.RandomBrightnessContrast(
                brightness_limit=config.data_config.brightness_limit,
                contrast_limit=config.data_config.contrast_limit,
                p=0.5,
            ),
            albu.GaussianBlur(blur_limit=3, p=0.5),
            albu.Normalize(),
            ToTensorV2(),
        ],
    )


def get_val_transform(image_width: int, image_height: int) -> albu.Compose:
    return albu.Compose(
        [
            albu.Resize(height=image_height, width=image_width),
            albu.Normalize(),
            ToTensorV2(),
        ],
    )
