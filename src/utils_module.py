from typing import Any

from torchmetrics import F1Score, MetricCollection, Precision, Recall


def get_metrics(**kwargs: Any) -> MetricCollection:
    return MetricCollection(
        {
            'f1': F1Score(**kwargs),
            'recall': Recall(**kwargs),
            'precision': Precision(**kwargs),
        },
    )
