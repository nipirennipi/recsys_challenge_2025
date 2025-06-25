import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

from training_pipeline.tasks import ValidTasks
from training_pipeline.metrics_containers import (
    MetricContainer,
)


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class MetricsAggregator:
    """
    Class for aggregating metrics collected during training.
    """

    def __init__(self, augmentation_method_hyperparameters: List[str | float]):
        self._aggregated_metrics: Dict[ValidTasks, List[MetricContainer]] = {}
        self.augmentation_method_hyperparameters = augmentation_method_hyperparameters

    def update(self, task: ValidTasks, metrics_tracker: List[MetricContainer]) -> None:
        """
        Method for attaching a metric tracker for aggregation later.
        """
        self._aggregated_metrics[task] = metrics_tracker

    def _find_best_weighted_metrics_and_epochs(self):
        """
        Method for determining max score and corresponding epoch from recorded scores.
        """

        def extract_weighted_metric(
            epoch_and_weighted_metric: Tuple[int, float]
        ) -> float:
            _, weighted_metric = epoch_and_weighted_metric
            return weighted_metric

        self._best_weighted_metrics: Dict[str, float] = {}
        self._best_epochs: Dict[str, int] = {}
        for task, metric_tracker in self._aggregated_metrics.items():
            weighted_metrics = [
                metric_container.compute_weighted_metric()
                for metric_container in metric_tracker
            ]
            best_epoch, best_weighted_metric = max(
                enumerate(weighted_metrics),
                key=extract_weighted_metric,
            )
            self._best_weighted_metrics[task.value] = best_weighted_metric
            self._best_epochs[task.value] = best_epoch

    def save(self, score_dir: Path):
        """
        Method that aggreagates the collected metrics, and saves them.
        """
        self._find_best_weighted_metrics_and_epochs()
        scores_fn = score_dir
        
        # hyperparameters = {
        #     "aug_method_1": self.augmentation_method_hyperparameters[0],
        #     "aug_method_2": self.augmentation_method_hyperparameters[1],
        #     "mask_proportion": float(self.augmentation_method_hyperparameters[2]),
        #     "crop_proportion": float(self.augmentation_method_hyperparameters[3]),
        #     "reorder_proportion": float(self.augmentation_method_hyperparameters[4]),
        # }
        result_to_log = {
            "timestamp": datetime.now().isoformat(),
            "best_metrics": self._best_weighted_metrics,
            "best_epochs": self._best_epochs,
            # "hyperparameters": hyperparameters,
        }
        
        with open(scores_fn, "a", encoding="utf-8") as scores_file:
            json.dump(result_to_log, scores_file, indent=4, ensure_ascii=False)
            scores_file.write("\n" + "=" * 50 + "\n")
        
        # logger.info(f"Augmentation methods 1: {self.augmentation_method_hyperparameters[0]}")
        # logger.info(f"Augmentation methods 2: {self.augmentation_method_hyperparameters[1]}")
        # logger.info(f"Mask proportion: {self.augmentation_method_hyperparameters[2]}")
        # logger.info(f"Crop proportion: {self.augmentation_method_hyperparameters[3]}")
        # logger.info(f"Reorder proportion: {self.augmentation_method_hyperparameters[4]}")
        logger.info(f"Best weighted metrics: {self._best_weighted_metrics}")
        