import time

import datasets
import hydra
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DefaultDataCollator

import dataset_utils
import wandb
from mamba_ssm_jax.modules.mamba_simple import MambaConfig, FlaxMambaLMHeadModel


def get_model_and_train_state(cfg, eos_token_id):
    config = MambaConfig(
        d_model=cfg.model.d_model,
        d_inner=cfg.model.d_inner,
        d_state=cfg.model.d_state,
        d_conv=cfg.model.d_conv,
        n_layer=cfg.model.n_layer,
        vocab_size=cfg.model.vocab_size,
        eos_token_id=eos_token_id,
    )
    model = FlaxMambaLMHeadModel(config)
    params = model.init_weights(jax.random.PRNGKey(42), (cfg.train.batch_size, cfg.model.seqlen))

    adamw = optax.adamw(cfg.train.learning_rate)
    state = train_state.TrainState.create(apply_fn=model.__call__, params=params, tx=adamw)
    return model, state


def flatten_norm(grads):
    flat_grads, _ = jax.tree_util.tree_flatten(grads)
    flat_grads = jnp.concatenate([jnp.reshape(g, -1) for g in flat_grads])
    return jnp.sqrt(jnp.sum(jnp.square(flat_grads))).item()


def train_step(cfg: DictConfig, state: train_state.TrainState, batch: jax.Array):
    """Trains one step."""
    B = batch.shape[0]
    mbs = cfg.train.micro_batch_size
    vocab_size = cfg.model.vocab_size
    assert B % mbs == 0, f"B={B}, mbs={mbs}"
    num_micro_batches = B // mbs

    def compute_loss(params, input_ids):
        inputs = input_ids[:, :-1]
        labels = input_ids[:, 1:]
        output = state.apply_fn(inputs, params=params)
        targets = jax.nn.one_hot(labels, num_classes=vocab_size)
        loss = optax.softmax_cross_entropy(output.logits, targets)
        return loss.mean(), output.logits

    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)

    accum_grads = None
    accum_loss = 0.0
    for idx in range(num_micro_batches):
        minibatch = batch[idx * mbs : (idx + 1) * mbs, :]
        (loss, _), grads = grad_fn(state.params, minibatch)
        grads = jax.tree_map(lambda x: x / num_micro_batches, grads)
        if accum_grads:
            grads = jax.tree_map(jnp.add, accum_grads, grads)
        accum_grads = grads
        accum_loss += loss

    grads = accum_grads
    loss = accum_loss / num_micro_batches

    state = state.apply_gradients(grads=grads)
    metrics = {
        "Train Loss": loss.mean().item(),
        "Grads Norm": flatten_norm(grads),
        "Parameters Norm": flatten_norm(state.params),
    }

    return state, metrics


def evaluate(cfg, test_dataloader, model):
    losses = []
    for batch in test_dataloader:
        input_ids = batch["input_ids"]
        inputs = input_ids[:, :-1]
        labels = input_ids[:, 1:]
        output = model(inputs)
        targets = jax.nn.one_hot(labels, num_classes=cfg.model.vocab_size)
        loss = optax.softmax_cross_entropy(output.logits, targets)

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
        collate_fn=DefaultDataCollator(return_tensors="np"),
        drop_last=True,
    )
    test_dataloader = DataLoader(
        dataset["test"],
        batch_size=1,
        collate_fn=DefaultDataCollator(return_tensors="np"),
    )

    # Prepare model and train state
    model, state = get_model_and_train_state(cfg, eos_token_id=tokenizer.eos_token_id)

    profile = cfg.train.get("profile", None)
    step_time_stack = []
    for idx, batch in enumerate(train_dataloader, start=1):
        if profile and idx == profile.start_step:
            jax.profiler.start_trace(**profile.trace_kwargs)

        input_ids = batch["input_ids"]
        start_time = time.time()
        state, metrics = train_step(cfg, state, input_ids)
        step_time_stack.append(time.time() - start_time)

        if idx % cfg.train.eval_interval == 0 or idx == len(train_dataloader):
            eval_loss = evaluate(cfg, test_dataloader, model)
            metrics["Validation Loss"] = eval_loss.item()

        if idx % cfg.train.log_interval == 0 or idx == len(train_dataloader):
            metrics["Training Step Time"] = sum(step_time_stack) / len(step_time_stack)
            step_time_stack = []
            wandb.log(metrics, step=idx)
            console.log(f"step-{idx}", metrics)

        if idx % cfg.train.save_interval == 0 or idx == len(train_dataloader):
            ckpt_dir = f"checkpoints/step-{idx}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

        if profile and idx == profile.end_step:
            jax.profiler.stop_trace()

    console.log(f"At the end of training, the metrics is:", metrics)


if __name__ == "__main__":
    train_and_evaluate_mamba()
