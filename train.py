from mamba_ssm_jax.modules.mamba_simple import Mamba, MambaLMHeadModel

import time

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from flax.training import train_state
import hydra
from omegaconf import DictConfig, OmegaConf
import datasets
import dataset_utils
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from einops import rearrange
import wandb
from rich.console import Console


def get_model(cfg):
    d_model = cfg.d_model
    vocab_size = cfg.vocab_size

    mamba_config = dict(
        d_model=d_model,
        d_inner=cfg.d_inner,
        d_state=cfg.d_state,
        d_conv=cfg.d_conv,
        bias=False,
        c_dt_rank="auto",
        conv_bias=True,
        use_fast_path=False,
    )
    block_config = dict(
        dim=d_model,
        mixer_cls=Mamba,
        mixer_kwargs=mamba_config,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
    )
    mixer_model_config = dict(
        dim=d_model,
        n_layer=cfg.n_layer,
        rms_norm=cfg.rms_norm,
        vocab_size=vocab_size,
        block_kwargs=block_config,
    )
    mamba_lm_head_model_config = dict(
        mixer_model_kwargs=mixer_model_config,
    )

    mamba_lm_head_model = MambaLMHeadModel(**mamba_lm_head_model_config)
    return mamba_lm_head_model


def get_initial_params(cfg, model, print_tabulate=True):
    seqlen = cfg.model.seqlen
    bs = cfg.train.batch_size

    key = jax.random.PRNGKey(42)
    variables = model.init(key, jnp.empty((bs, seqlen), dtype=jnp.int32))
    if print_tabulate:
        model_info = model.tabulate(key, jnp.empty((bs, seqlen), dtype=jnp.int32), compute_flops=False, compute_vjp_flops=False)
        print(model_info)

    return variables['params']


def get_model_and_train_state(cfg):
    model = get_model(cfg.model)
    params = get_initial_params(cfg, model)
    if cfg.model.bf16:
        params = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)

    tx = optax.adamw(cfg.train.learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx,
    )
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
    assert B % mbs == 0
    num_micro_batches = B // mbs

    def compute_loss(params, input_ids):
        inputs = input_ids[:, :-1]
        labels = input_ids[:, 1:]
        output = state.apply_fn(
            {'params': params},
            inputs,
        )
        targets = jax.nn.one_hot(labels, num_classes=vocab_size)
        loss = optax.softmax_cross_entropy(output.logits, targets)
        return loss.mean(), output.logits
    
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)

    accum_grads = None
    accum_loss = 0.
    for idx in range(num_micro_batches):
        minibatch = batch[idx * mbs: (idx + 1) * mbs, :]
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


def evaluate(cfg, test_dataloader, model, params):
    losses = []
    for batch in test_dataloader:
        input_ids = [x.numpy() for x in batch["input_ids"]]
        input_ids = rearrange(jnp.array(input_ids), "L B -> B L")
        inputs = input_ids[:, :-1]
        labels = input_ids[:, 1:]
        output = model.apply({'params': params}, inputs)
        targets = jax.nn.one_hot(labels, num_classes=cfg.model.vocab_size)
        loss = optax.softmax_cross_entropy(output.logits, targets)

        losses.append(loss.mean())

    return sum(losses) / len(losses)


@hydra.main(version_base="1.1", config_path="configs", config_name="mamba-param_3M-d_state_32-d_conv_8-seqlen_64.yaml")
def train_and_evaluate_mamba(cfg : DictConfig):
    wandb.init(project="mamba", config=OmegaConf.to_container(cfg))
    console = Console()

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.path)
    dataset = datasets.load_dataset(cfg.data.path)

    dataset = dataset_utils.prepare_dataset(
        raw_dataset=dataset,
        tokenizer=tokenizer,
        seq_length=cfg.model.seqlen,
    )

    train_dataloader = DataLoader(
        dataset["train"], batch_size=cfg.train.batch_size,
    )
    test_dataloader = DataLoader(dataset["test"], batch_size=1)

    step_time_stack = []
    model, state = get_model_and_train_state(cfg)
    for idx, batch in enumerate(train_dataloader, start=1):
        input_ids = [x.numpy() for x in batch["input_ids"]]
        input_ids = rearrange(jnp.array(input_ids), "L B -> B L")
        start_time = time.time()
        state, metrics = train_step(cfg, state, input_ids)
        step_time_stack.append(time.time() - start_time)
        if idx % cfg.train.log_interval == 0:
            eval_loss = evaluate(cfg, test_dataloader, model, state.params)
            metrics["Validation Loss"] = eval_loss.item()
            metrics["Training Step Time"] = sum(step_time_stack) / len(step_time_stack)
            step_time_stack = []
            wandb.log(metrics, step=idx)
            console.log(f"step-{idx}", metrics, time.time())


if __name__ == "__main__":
    train_and_evaluate_mamba()
