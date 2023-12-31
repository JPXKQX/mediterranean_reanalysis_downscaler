import os
from inspect import signature
from typing import Dict

import matplotlib.pyplot
import numpy as np
import torch
from accelerate import Accelerator, find_executable_batch_size
from accelerate.utils import LoggerType
from transformers import get_cosine_schedule_with_warmup
from huggingface_hub import Repository
from tqdm import tqdm
import logging
from reanalysis_downscaler.dataset.generator import DataGenerator
from reanalysis_downscaler.model.configs import TrainingConfig
from reanalysis_downscaler.model.loss import compute_loss

logger = getLogger(__name__)

model_kwargs = {"return_dict": False}


def train_nn(
    config: TrainingConfig,
    model,
    train_dataset: DataGenerator,
    val_dataset: DataGenerator,
    hparams: Dict = {},
):
    """
    Train a neural network model.

    Parameters
    ----------
    config : TrainingConfig
        The training configuration.
    model : nn.Module
        The neural network model.
    train_dataset : DataGenerator
        The training dataset.
    val_dataset : DataGenerator
        The validation dataset.
    hparams : Dict, optional
        Hyperparameters.

    Returns
    -------
    model : nn.Module
        The trained model.
    repo_name : str
        The repository name.

    Notes
    -----
    This function performs the training of a neural network model using the provided
    datasets and configuration.
    """
    hparams = hparams | config.__dict__
    hparams.pop("__pydantic_initialised__", None)
    number_model_params = int(sum([np.prod(m.size()) for m in model.parameters()]))
    if "number_model_params" not in hparams:
        hparams["number_model_params"] = number_model_params

    model_name = model.__class__.__name__
    run_name = f"Train Super-Resolution NN ({model_name})"
    accelerator = Accelerator(
        cpu=config.device == "cpu",
        device_placement=True,
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=[LoggerType.TENSORBOARD],
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    accelerator.free_memory()  # Free all lingering references

    # Define important objects
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, pin_memory=True)
    dataloader_val = torch.utils.data.DataLoader(
        val_dataset, batch_size, pin_memory=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(dataloader) * config.num_epochs),
    )

    if accelerator.is_main_process:
        if config.push_to_hub:
            repo = Repository(
                config.output_dir,
                clone_from=config.hf_repo_name,
                token=os.getenv("HF_TOKEN"),
            )
            repo.git_pull()
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers(run_name, config=hparams)
        tfboard_tracker = accelerator.get_tracker("tensorboard")

    (
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, dataloader, dataloader_val, lr_scheduler
    )

    # Get fixed samples
    val_era5, val_cerra = next(iter(val_dataloader))
    if batch_size > 4:
        val_era5, val_cerra = val_era5[:4], val_cerra[:4]

    # tfboard_tracker.writer.add_graph(model, val_era5)
    logger.info(f"Number of parameters: {number_model_params}")
    global_step = 0
    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader) + len(val_dataloader),
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch+1}")

        for era5, cerra in train_dataloader:
            # Predict the noise residual
            with accelerator.accumulate(model):
                cerra_pred = model(era5, **model_kwargs)[0]
                l1, l_lowres, l_blurred = compute_loss(cerra_pred, cerra)
                loss = l1 + l_lowres + l_blurred
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            cerra_pred_base = torch.nn.functional.interpolate(
                era5[..., 6:-6, 6:-6], scale_factor=5, mode="bicubic"
            )
            l1_base, l_lowres_base, l_blurred_base = compute_loss(
                cerra_pred_base, cerra
            )
            loss_base = l1_base + l_lowres_base + l_blurred_base
            progress_bar.update(1)
            pred_var = cerra_pred.var(keepdim=True, dim=0).mean().item()
            true_var = cerra.var(keepdim=True, dim=0).mean().item()
            lo = loss.detach().item()
            l_base = loss_base.detach().item()
            logs = {
                "loss_vs_step": lo,
                "l1_pred_vs_step": l1.mean().item(),
                "l1_lowres_vs_step": l_lowres.mean().item(),
                "l1_blurred_vs_step": l_blurred.mean().item(),
                "baseline_loss_vs_step": l_base,
                "improvement_vs_step": (l_base - lo) / l_base * 100,
                "lr_vs_step": lr_scheduler.get_last_lr()[0],
                "step": global_step,
                "bias_perc_vs_step": (cerra - cerra_pred).mean().item()
                / cerra.mean().item(),
                "mean_var_ratio_vs_step": true_var / pred_var,
                "epoch": epoch,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            tfboard_tracker.writer.add_histogram(
                "cerra prediction", cerra_pred, global_step
            )
            tfboard_tracker.writer.add_histogram("cerra", cerra, global_step)
            global_step += 1

        # Evaluate
        loss, l1_pred, l1_lowres, l1_blurred = [], [], [], []
        true_var, pred_var, bias, mean_pred = [], [], [], []
        for era5, cerra in val_dataloader:
            # Predict the noise residual
            with torch.no_grad():
                cerra_pred = model(era5, **model_kwargs)[0]
                l_pred, l_lowres, l_blurred = compute_loss(cerra_pred, cerra)
                loss.append((l_pred + l_lowres + l_blurred).mean().item())
                l1_pred.append(l_pred.mean().item())
                l1_lowres.append(l_lowres.mean().item())
                l1_blurred.append(l_blurred.mean().item())

            pred_var.append(cerra_pred.var(keepdim=True, dim=0).mean().item())
            true_var.append(cerra.var(keepdim=True, dim=0).mean().item())
            bias.append((cerra - cerra_pred).mean().item())
            mean_pred.append(cerra_pred.mean().item())
            progress_bar.update(1)

        logs = {
            "val_loss_vs_epoch": sum(loss) / len(loss),
            "val_l1_vs_epoch": sum(l1_pred) / len(l1_pred),
            "val_l1_lowres_vs_epoch": sum(l1_lowres) / len(l1_lowres),
            "val_l1_blurred_vs_epoch": sum(l1_blurred) / len(l1_blurred),
            "val_bias_perc_vs_epoch": sum(bias) / sum(mean_pred),
            "val_mean_var_ratio_vs_epoch": sum(true_var) / sum(pred_var),
            "epoch": epoch,
        }
        accelerator.log(logs, step=epoch)
        progress_bar.close()

        # After each epoch you optionally sample some demo images
        if accelerator.is_main_process:
            is_last_epoch = epoch == config.num_epochs - 1

            if (epoch + 1) % config.save_model_epochs == 0 or is_last_epoch:
                logger.info("Saving model weights...")
                model.save_pretrained(config.output_dir)
                if config.push_to_hub:
                    repo.push_to_hub(
                        commit_message=f"Epoch {epoch+1}", blocking=True
                    )

    accelerator.end_training()

    return model, config.hf_repo_name
