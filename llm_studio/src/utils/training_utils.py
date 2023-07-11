import logging
import os
from typing import Any, Dict

import coolname
import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from llm_studio.src.optimizers import Optimizers
from llm_studio.src.schedulers import Schedulers
from llm_studio.src.utils.data_utils import cat_batches, get_inference_batch_size
from llm_studio.src.utils.exceptions import LLMDataException, LLMModelException
from llm_studio.src.utils.logging_utils import TqdmToLogger
from llm_studio.src.utils.modeling_utils import logger, unwrap_model
from llm_studio.src.utils.utils import save_pickle


def save_checkpoint(model: torch.nn.Module, path: str, cfg: Any):
    """Saves a model checkpoint if the path is provided.

    Args:
        model: model to save
        path: path to save the checkpoint to

    Returns:
        Dictionary with all the keys to save
    """

    model = unwrap_model(model)

    if hasattr(cfg.training, "lora") and cfg.training.lora:
        model.backbone.save_pretrained(path)

    checkpoint = {"model": model.state_dict()}

    if path is not None:
        torch.save(checkpoint, os.path.join(path, "checkpoint.pth"))


def get_optimizer(model: torch.nn.Module, cfg: Any) -> torch.optim.Optimizer:
    """Prepares Optimizer.

    Args:
        model: model
        cfg: input config

    Returns:
        Optimizer
    """

    no_decay = ["bias", "LayerNorm.weight"]
    differential_layers = cfg.training.differential_learning_rate_layers
    optimizer = Optimizers.get(cfg.training.optimizer)(
        [
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (not any(layer in name for layer in differential_layers))
                    and (not any(nd in name for nd in no_decay))
                    # and param.requires_grad
                ],
                "lr": cfg.training.learning_rate,
                "weight_decay": cfg.training.weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (not any(layer in name for layer in differential_layers))
                    and (any(nd in name for nd in no_decay))
                    # and param.requires_grad
                ],
                "lr": cfg.training.learning_rate,
                "weight_decay": 0,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (any(layer in name for layer in differential_layers))
                    and (not any(nd in name for nd in no_decay))
                    # and param.requires_grad
                ],
                "lr": cfg.training.differential_learning_rate,
                "weight_decay": cfg.training.weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (any(layer in name for layer in differential_layers))
                    and (any(nd in name for nd in no_decay))
                    # and param.requires_grad
                ],
                "lr": cfg.training.differential_learning_rate,
                "weight_decay": 0,
            },
        ],
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    return optimizer


def get_scheduler(
    cfg: Any, optimizer: torch.optim.Optimizer, epoch_steps: int
) -> torch.optim.lr_scheduler._LRScheduler:
    """Prepares Learning Rate Scheduler.

    Args:
        cfg: input config
        optimizer: model optimizer
        epoch_steps: total number of weight updates during the epoch

    Returns:
        Learning Rate Scheduler
    """

    scheduler = Schedulers.get(cfg.training.schedule)(
        optimizer=optimizer,
        num_warmup_steps=cfg.training.warmup_epochs * epoch_steps,
        num_training_steps=cfg.training.epochs * epoch_steps,
    )

    return scheduler


def generate_experiment_name() -> str:
    """
    Generates a random human-readable experiment name in kebab-case.

    Returns:
        The random name.
    """
    return coolname.generate_slug(2)


def reduce_metric(output, reduce=None) -> float:
    """Reduces metric and return metric score (number)

    Args:
        output: output of the model
        reduce: how to reduce the metric over the sample dimension

    Returns:
        score: single number score (using config threshold for threshold metrics)
        or non-reduced array of scores per sample.
    """

    if reduce == "mean":
        score = np.mean(output["metrics"])
    else:
        raise NotImplementedError()

    return score


def get_number_of_validation_epochs(training_epochs: int, evaluation_epochs: float):
    """
    Given the number of training epochs and the number of epochs between model
    evaluations, return the number of times the model is being evaluated during
    training

    Args:
        training_epochs: The number of epochs to train for
        evaluation_epochs: This is the number of epochs after which we want to
            evaluate our model

    Returns:
        num_val_epochs: The number of epochs to be evaluated during training.
    """
    return training_epochs // evaluation_epochs


def contains_nan(output: Dict):
    return (
        sum(
            [
                1
                for key, val in output.items()
                if isinstance(val, torch.Tensor)
                and torch.isnan(val.detach().cpu()).sum() > 0
            ]
        )
        > 0
    )


def run_inference(
    cfg: Any,
    model: torch.nn.Module,
    dataloader,
    mode: str,
) -> Dict[str, list]:
    """Runs inference

    Args:
        cfg: config
        model: model
        dataloader: custom dataloader
        mode: mode for inference

    Returns:
        Dictionary with output

    """

    # Store information for evaluation
    out = dict()

    if cfg.environment._local_rank == 0:
        logger.info(f"Starting {mode} inference")

    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    progress_bar = tqdm(
        total=len(dataloader),
        disable=cfg.environment._local_rank != 0,
        file=tqdm_out,
        ascii=True,
        desc=f"{mode} progress",
        mininterval=0,
    )

    log_update_steps = max(len(dataloader) // 20, 1)
    inf_it = iter(dataloader)
    for itr in range(len(dataloader)):
        try:
            data = next(inf_it)
        except Exception:
            raise LLMDataException("Data reading error. Skipping inference.")

        val_batch_size = get_inference_batch_size(cfg)
        cfg.environment._curr_val_step += val_batch_size * cfg.environment._world_size

        batch = cfg.dataset.dataset_class.batch_to_device(data, cfg.environment._device)

        with autocast(enabled=cfg.environment.mixed_precision):
            output = model.forward(batch)
            if cfg.prediction.metric != "Perplexity":
                output["predicted_answer_ids"] = (
                    unwrap_model(model).generate(batch, cfg).detach().cpu()
                )
        if contains_nan(output) and cfg.environment.mixed_precision:
            raise LLMModelException(
                "NaN caught during mixed precision inference. "
                "Please disable mixed precision inference. "
                "Alternatively, reducing learning rate or "
                "gradient clipping may help to stabilize training."
            )

        output = dataloader.dataset.postprocess_batch_predictions(
            cfg=cfg, output=output
        )

        if "predicted_answer_ids" in output.keys():
            del output["predicted_answer_ids"]

        for key, val in output.items():
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu()

            # DefaultDict is not used as it adds extra keys during pickle.dump
            if key not in out:
                out[key] = [val]
            else:
                out[key] += [val]

        if cfg.environment._local_rank == 0:
            # Show logs each 5% of the inference
            if (itr + 1) % log_update_steps == 0 or itr == len(dataloader) - 1:
                progress_bar.set_description(f"{mode} progress", refresh=False)
                if (itr + 1) % log_update_steps == 0:
                    progress_bar.update(log_update_steps)
                else:
                    progress_bar.update(len(dataloader) % log_update_steps)

            cfg.logging._logger.log(
                "internal",
                "current_val_step",
                cfg.environment._curr_val_step,
                step=cfg.environment._curr_val_step,
            )

        if cfg.environment._distributed:
            torch.distributed.barrier()

    progress_bar.close()
    del progress_bar
    out = cat_batches(out)

    return out


def save_predictions(cfg, val_data, val_dataloader, val_df, mode):
    val_data, val_df = val_dataloader.dataset.format_output(  # type: ignore
        cfg=cfg, df=val_df, output=val_data
    )
    raw_preds_name = os.path.join(cfg.output_directory, f"{mode}_raw_predictions.pkl")
    csv_preds_name = os.path.join(cfg.output_directory, f"{mode}_predictions.csv")
    save_pickle(raw_preds_name, val_data)
    val_df.to_csv(csv_preds_name, index=False)
