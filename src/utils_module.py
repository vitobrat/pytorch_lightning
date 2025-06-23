from typing import Any

from torchmetrics import F1Score, MetricCollection, Precision, Recall


def get_metrics(**kwargs: Any) -> MetricCollection:
    return MetricCollection(
        {
            F1Score: Precision(**kwargs),
            Recall: Recall(**kwargs),
            Precision: Precision(**kwargs),
        },
    )
