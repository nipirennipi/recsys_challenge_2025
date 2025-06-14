import logging
import numpy as np
import pytorch_lightning as pl

from pathlib import Path
from typing import List
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import NeptuneLogger

from data_utils.data_dir import DataDir
from validator.validate import (
    validate_and_load_embeddings,
)
from multi_task.logger_factory import (
    NeptuneLoggerFactory,
)
from multi_task.model import (
    UniversalModel,
)
from multi_task.tasks import ValidTasks
from multi_task.data_module import (
    BehavioralDataModule,
)
from multi_task.constants import (
    GPU_MEMORY_TO_ALLOCATE,
    BATCH_SIZE,
    MAX_EMBEDDING_DIM,
    EMBEDDING_DIM,
    HIDDEN_SIZE_THIN,
    HIDDEN_SIZE_WIDE,
    LEARNING_RATE,
    MAX_EPOCH,
)
from multi_task.target_data import (
    TargetData,
)
from multi_task.task_constructor import (
    TaskConstructor,
    TaskSettings,
    transform_client_ids,
)
from multi_task.metric_aggregator import (
    MetricsAggregator,
)
from multi_task.preprocess_data import (
    IdMapper,
    InfrequentUrlRecorder,
)
from data_utils.constants import (
    EventTypes,
)
from multi_task.gpu_allocator import (
    GPUAllocator,
)


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def run_training(
    tasks: List[ValidTasks],
    task_constructor: TaskConstructor,
    data_dir: DataDir,
    id_mapper: IdMapper,
    target_data: TargetData,
    num_workers: int,
    accelerator: str,
    devices: List[int] | str | int,
    neptune_logger: NeptuneLogger,
) -> None:
    """
    Function for running the training of a model, with all the training
    parameters already established.

    Args:
        task_settings (TaskSettings): Settings for running the task
        embeddings (np.ndarray): Embeddings to be used as the input to the model
        client_ids (np.ndarray): The ids of clients, in order as their embeddings are in `embeddings`.
        target_data (TargetData): Target purchase data based on which targets are computed
        num_workers (int): Number of workers to be used for loading data
        accelerator (str): Type of device to run training on (e.g. gpu, cpu, etc.)
        devices (List[int] | str | int): id of devices used for training
        neptune_logger (NeptuneLogger): logger instance where training information is logged
    """
    
    gpu_allocator = GPUAllocator(GPU_MEMORY_TO_ALLOCATE, devices)
    gpu_allocator.allocate_gpu_memory()

    task_settings = [
        task_constructor.construct_task(task=task) for task in tasks
    ]
    target_calculators = [
        task_setting.target_calculator for task_setting in task_settings
    ]

    data = BehavioralDataModule(
        data_dir=data_dir,
        id_mapper=id_mapper,
        target_data=target_data,
        target_calculators=target_calculators,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        gpu_allocator=gpu_allocator,
    )

    sku_vocab_size = id_mapper.sku_vocab_size()
    category_vocab_size = id_mapper.category_vocab_size()
    event_type_vocab_size = len(EventTypes)
    url_vocab_size = id_mapper.url_vocab_size()
    item_stat_feat_dim = data.item_features_dim
    loss_fn = [
        task_setting.loss_fn for task_setting in task_settings
    ]
    out_put_dims = [
        target_calculator.target_dim for target_calculator in target_calculators
    ]

    model = UniversalModel(
        sku_vocab_size=sku_vocab_size,
        category_vocab_size=category_vocab_size,
        event_type_vocab_size=event_type_vocab_size,
        url_vocab_size=url_vocab_size,
        item_stat_feat_dim=item_stat_feat_dim,
        output_dims=out_put_dims,
        hidden_size_thin=HIDDEN_SIZE_THIN,
        hidden_size_wide=HIDDEN_SIZE_WIDE,
        learning_rate=LEARNING_RATE,
        # metric_calculator=task_settings.metric_calculator,
        loss_fn=loss_fn,
        # metrics_tracker=task_settings.metrics_tracker,
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=MAX_EPOCH,
        logger=neptune_logger,
        callbacks=RichProgressBar(leave=True),
        log_every_n_steps=5000,
        check_val_every_n_epoch=MAX_EPOCH,
        enable_checkpointing=False,
    )

    trainer.fit(model=model, datamodule=data)


def run_tasks(
    neptune_logger_factory: NeptuneLoggerFactory,
    tasks: List[ValidTasks],
    task_constructor: TaskConstructor,
    data_dir: DataDir,
    num_workers: int,
    accelerator: str,
    devices: List[int] | str | int,
    score_dir: Path | None,
    disable_relevant_clients_check: bool,
) -> None:
    """
    Function for running a task, i.e. setting up the training, and the starting the training. This method first
    prepares running paramteres based on preliminary setup, and the calls the `run_train` method.

    Args:
        neptune_logger_factory (NeptuneLoggerFactory): Factory that can generate instance of neptune loggers
            with some pre-set parameters common to the embedding on which experiments are run.
        tasks (List[ValidTasks]): tasks on which the embeddings are to be evaluated.
        task_constructor (TaskConstructor): object for generating training settings based on the task
        data_dir (DataDir): container for simplified access to subdirectories of data_dir.
        embeddings_dir (Path): Path to the directory where the embeddings are stored.
        num_workers (int): number of workers to be used for loading data
        accelerator (str): Type of device to run training on (e.g. gpu, cpu, etc.)
        devices (List[int] | str | int): id of devices used for training
        score_dir (Path | None): Path where results are saved in an easy-to-read format, parallel to netune logging.
        disable_relevant_clients_check (bool): disables validator check for relevant clients
    """
    target_data = TargetData.read_from_dir(target_dir=data_dir.target_dir)
    metrics_aggregator = MetricsAggregator()
    logger.info("Running on multi_tasks")

    id_mapper = IdMapper(challenge_data_dir=data_dir)
    id_mapper.load_mapping()
    
    # Cutoff infrequent URLs
    infrequent_url_recorder = InfrequentUrlRecorder(challenge_data_dir=data_dir)
    infrequent_url_recorder.load_infrequent_url()
    id_mapper.cutoff_infrequent_url(
        infrequent_urls=infrequent_url_recorder.infrequent_url
    )
    
    logger.info("Setting up training logger")
    neptune_logger = neptune_logger_factory.get_logger()

    logger.info("Running training")
    run_training(
        tasks=tasks,
        task_constructor=task_constructor,
        data_dir=data_dir,
        id_mapper=id_mapper,
        target_data=target_data,
        num_workers=num_workers,
        accelerator=accelerator,
        devices=devices,
        neptune_logger=neptune_logger,
    )
    neptune_logger.experiment.stop()

    # metrics_aggregator.update(
    #     task=task, metrics_tracker=task_settings.metrics_tracker
    # )
    logger.info("Run on multi_tasks completed")

    if score_dir:
        metrics_aggregator.save(score_dir=score_dir)
