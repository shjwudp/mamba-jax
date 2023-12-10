from mamba_ssm_jax.modules.mamba_simple import Mamba, MambaLMHeadModel

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

    key = jax.random.key(0)
    variables = model.init(key, jnp.empty((bs, seqlen), dtype=jnp.int32))
    if print_tabulate:
        model_info = model.tabulate(key, jnp.empty((bs, seqlen), dtype=jnp.int32), compute_flops=True, compute_vjp_flops=True)
        print(model_info)

    return variables['params']


def get_model_and_train_state(cfg):
    model = get_model(cfg.model)
    params = get_initial_params(cfg, model)
    if cfg.model.bf16:
        params = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)

    tx = optax.adamw(cfg.train.learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )
    return model, state

def grad_norm(grads):
    flat_grads, _ = jax.tree_util.tree_flatten(grads)
    flat_grads = jnp.concatenate([jnp.reshape(g, -1) for g in flat_grads])
    return jnp.sqrt(jnp.sum(jnp.square(flat_grads))).item()


def train_step(cfg : DictConfig, state: train_state.TrainState, input_ids: jax.Array):
    """Trains one step."""
    inputs = input_ids[:, :-1]
    labels = input_ids[:, 1:]

    def loss_fn(params):
        output = state.apply_fn(
            {'params': params},
            inputs,
        )
        targets = jax.nn.one_hot(labels, num_classes=cfg.model.vocab_size)
        loss = optax.softmax_cross_entropy(output.logits, targets)
        return loss.mean(), output.logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = {
        "Train Loss": loss.item(),
        "Grads Norm": grad_norm(grads),
        "Parameters Norm": grad_norm(state.params),
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


@hydra.main(version_base="1.1", config_path="configs", config_name="config.yaml")
def train_and_evaluate_mamba(cfg : DictConfig):
    wandb.init(project="mamba", config=OmegaConf.to_container(cfg))

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

    model, state = get_model_and_train_state(cfg)
    for idx, batch in enumerate(train_dataloader, start=1):
        input_ids = [x.numpy() for x in batch["input_ids"]]
        input_ids = rearrange(jnp.array(input_ids), "L B -> B L")
        state, metrics = train_step(cfg, state, input_ids, )
        if idx % cfg.train.log_interval == 0:
            eval_loss = evaluate(cfg, test_dataloader, model, state.params)
            metrics["Validation Loss"] = eval_loss.item()
            wandb.log(metrics, step=idx)
            print(f"step-{idx}", metrics)


if __name__ == "__main__":
    train_and_evaluate_mamba()
