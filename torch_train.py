import time

import datasets
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer, DefaultDataCollator
from einops import rearrange

import dataset_utils
import wandb
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig


def get_model(cfg):
    config = MambaConfig(
        d_model=cfg.model.d_model,
        n_layer=cfg.model.n_layer,
        vocab_size=cfg.model.vocab_size,
        ssm_cfg=dict(
            d_state=cfg.model.d_state,
            d_conv=cfg.model.d_conv,
        ),
    )
    model = MambaLMHeadModel(config)

    return model


def flatten_norm(grads):
    flat_grads, _ = jax.tree_util.tree_flatten(grads)
    flat_grads = jnp.concatenate([jnp.reshape(g, -1) for g in flat_grads])
    return jnp.sqrt(jnp.sum(jnp.square(flat_grads))).item()


def get_gradient_norm(model):
    max_norm = float('inf')
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=2.0)
    return total_norm


def calculate_norm_efficient(items, norm_type=2):
    items = [item.detach().flatten() for item in items]
    total_norm = torch.norm(torch.cat(items), norm_type)
    return total_norm.item()


def train_step(cfg: DictConfig, model: torch.nn.Module, batch: torch.Tensor):
    """Trains one step."""
    B = batch.shape[0]
    mbs = cfg.train.micro_batch_size
    vocab_size = cfg.model.vocab_size
    assert B % mbs == 0, f"B={B}, mbs={mbs}"
    num_micro_batches = B // mbs

    def compute_loss(model, input_ids):
        inputs = input_ids[:, :-1]
        labels = input_ids[:, 1:]
        output = model(inputs)
        loss = F.cross_entropy(
            rearrange(output.logits, "b s v -> (b s) v"),
            rearrange(labels, "b s -> (b s)"),
        )

        return loss

    accum_loss = 0.0
    for idx in range(num_micro_batches):
        minibatch = batch[idx * mbs : (idx + 1) * mbs, :]

        loss = compute_loss(model, minibatch)
        accum_loss += loss

    loss = accum_loss / num_micro_batches
    loss.backward()

    metrics = {
        "Train Loss": loss.mean().item(),
        "Grads Norm": calculate_norm_efficient([p.grad for p in model.parameters() if p.grad is not None]),
        "Parameters Norm": calculate_norm_efficient([p.data for p in model.parameters()]),
    }

    return metrics


def evaluate(cfg, test_dataloader, model):
    losses = []
    for batch in test_dataloader:
        input_ids = batch["input_ids"]
        inputs = input_ids[:, :-1]
        labels = input_ids[:, 1:]
        output = model(inputs)
        loss = F.cross_entropy(
            rearrange(output.logits, "b s v -> (b s) v"),
            rearrange(labels, "b s -> (b s)"),
        )

        losses.append(loss.mean())

    return sum(losses) / len(losses)


@hydra.main(
    version_base="1.1",
    config_path="configs",
    config_name="mamba-param_3M-d_state_32-d_conv_8-seqlen_64.yaml",
)
def train_and_evaluate_mamba(cfg: DictConfig):
    wandb.init(project="mamba", config=OmegaConf.to_container(cfg))
    console = Console()

    # Prepare dataset and dataloader
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.path)
    dataset = datasets.load_dataset(cfg.data.path)
    dataset = dataset_utils.prepare_dataset(
        raw_dataset=dataset, tokenizer=tokenizer, seq_length=cfg.model.seqlen,
    )
    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=cfg.train.batch_size,
        collate_fn=DefaultDataCollator(return_tensors="pt"),
        drop_last=True,
    )
    test_dataloader = DataLoader(
        dataset["test"],
        batch_size=1,
        collate_fn=DefaultDataCollator(return_tensors="pt"),
    )

    # Prepare model and train state
    model = get_model(cfg)

    step_time_stack = []
    for idx, batch in enumerate(train_dataloader, start=1):
        input_ids = batch["input_ids"]
        start_time = time.time()
        metrics = train_step(cfg, model, input_ids)
        step_time_stack.append(time.time() - start_time)

        if idx % cfg.train.log_interval == 0 or idx == len(train_dataloader):
            eval_loss = evaluate(cfg, test_dataloader, model)
            metrics["Validation Loss"] = eval_loss.item()
            metrics["Training Step Time"] = sum(step_time_stack) / len(step_time_stack)
            step_time_stack = []
            wandb.log(metrics, step=idx)
            console.log(f"step-{idx}", metrics)

    console.log(f"At the end of training, the metrics is:", metrics)


if __name__ == "__main__":
    torch.set_default_device('cuda')
    train_and_evaluate_mamba()
