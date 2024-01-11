import math
from collections import namedtuple
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from einops import rearrange, repeat
from flax import linen as nn
from transformers import FlaxPreTrainedModel, PretrainedConfig


class MambaConfig(PretrainedConfig):
    model_type = "mamba"
    attribute_map = {
        "hidden_size": "d_model",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        d_model=64,
        d_inner=128,
        d_state=32,
        d_conv=8,
        n_layer=2,
        vocab_size=50257,
        **kwargs,
    ):
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv = d_conv
        self.n_layer = n_layer
        self.vocab_size = vocab_size

        super().__init__(**kwargs)


class Mamba(nn.Module):
    config: MambaConfig
    bias: bool = False
    c_dt_rank: str = "auto"
    conv_bias: bool = True
    use_fast_path: bool = False

    def setup(self):
        d_model = self.config.d_model
        d_inner = self.config.d_inner
        d_state = self.config.d_state
        d_conv = self.config.d_conv

        self.dt_rank = (
            math.ceil(d_model / 16) if self.c_dt_rank == "auto" else self.c_dt_rank
        )
        self.in_proj = nn.Dense(d_inner * 2, use_bias=self.bias)
        self.conv1d = nn.Conv(
            features=d_inner,
            use_bias=self.conv_bias,
            kernel_size=(d_conv,),
            feature_group_count=d_inner,
            padding="SAME",
        )
        self.x_proj = nn.Dense(self.dt_rank + d_state * 2, use_bias=False)
        self.dt_proj = nn.Dense(d_inner, use_bias=True)

        # S4D real initialization
        A = repeat(
            jnp.arange(1, d_state + 1, dtype=jnp.float32),
            "n -> d n",
            d=d_inner,
        )
        self.A_log = jnp.log(A)

        # D "skip" parameter
        self.D = jnp.ones(d_inner)

        self.out_proj = nn.Dense(
            d_model,
            use_bias=self.bias,
            kernel_init=nn.initializers.kaiming_uniform(),
        )

    @nn.compact
    def __call__(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        d_state = self.config.d_state
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None

        xz = self.in_proj(hidden_states)

        A = -jnp.exp(self.A_log.astype(jnp.float32))  # (d_inner, d_state)
        assert not self.use_fast_path
        xz = rearrange(xz, "b l (d i) -> i b l d", i=2)
        x, z = xz[0], xz[1]
        x = self.conv1d(x)
        x = nn.activation.silu(x)

        x_dbl = self.x_proj(x)  # (b l d)
        dt, B, C = jnp.split(
            x_dbl, [self.dt_rank, self.dt_rank + d_state], axis=-1
        )
        dt = self.dt_proj(dt)
        x = rearrange(x, "b l d -> b d l")
        dt = rearrange(dt, "b l d -> b d l")
        B = rearrange(B, "b l n -> b n l")
        C = rearrange(C, "b l n -> b n l")
        z = rearrange(z, "b l d -> b d l")
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.astype(jnp.float32),
            z=z,
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        y = rearrange(y, "b d l -> b l d")
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        y = y.astype(hidden_states.dtype)
        out = self.out_proj(y)

        return out


class Block(nn.Module):
    config: MambaConfig
    mixer_cls: Callable = Mamba
    norm_cls: Callable = nn.LayerNorm
    fused_add_norm: bool = False
    residual_in_fp32: bool = False

    def setup(self):
        self.mixer = self.mixer_cls(self.config)
        self.norm = self.norm_cls(self.config.d_model)

    @nn.compact
    def __call__(self, hidden_states, residual=None, inference_params=None):
        """
        hidden_states: (B, L, D)
        residual: (B, L, D)
        Returns: same shape as hidden_states
        """
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.astype(hidden_states.dtype))
        if self.residual_in_fp32:
            residual = residual.astype(jnp.float32)

        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual


class MixerModel(nn.Module):
    config: MambaConfig
    rms_norm: bool = False

    def setup(self):
        dim = self.config.d_model
        n_layer = self.config.n_layer
        vocab_size = self.config.vocab_size

        self.embedding = nn.Embed(vocab_size, dim)
        self.layers = [Block(self.config) for i in range(n_layer)]
        self.norm_f = (
            nn.LayerNorm(dim) if not self.rms_norm else nn.RMSNorm(dim)
        )

    def __call__(self, input_ids, params: dict = None):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.astype(hidden_states.dtype))
        return hidden_states


class MambaLMHeadModel(nn.Module):
    config: MambaConfig

    def setup(self):
        self.backbone = MixerModel(self.config)

    def __call__(self, input_ids):
        hidden_states = self.backbone(input_ids)
        lm_logits = self.backbone.embedding.attend(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)


class FlaxMambaLMHeadModel(FlaxPreTrainedModel):
    config_class = MambaConfig
    module_class: nn.Module = MambaLMHeadModel

    def __init__(
        self,
        config: MambaConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
    ):
        module = self.module_class(config)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple):
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        module_init_outputs = self.module.init(rngs, jnp.empty(input_shape, dtype=jnp.int32))
        return module_init_outputs["params"]

    def __call__(
        self,
        input_ids,
        params: dict = None,
    ):
        outputs = self.module.apply({"params": params or self.params}, input_ids)

        return outputs


def view_as_complex(arr):
    # Ensure the last dimension is of size 2
    assert arr.shape[-1] == 2, "Last dimension must be of size 2"

    # Split the last dimension into real and imaginary parts
    real, imag = jnp.split(arr, 2, axis=-1)

    # Create a complex array
    complex_array = real.squeeze(-1) + 1j * imag.squeeze(-1)

    return complex_array


def selective_scan_fn(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)

    out: r(B D L)
    last_state (optional): r(B D dstate)
    """
    # TODO: support boundary resetting mentioned in Mamba paper 3.5.2
    dtype_in = u.dtype
    u = u.astype(jnp.float32)
    delta = delta.astype(jnp.float32)

    if delta_softplus:
        delta = jax.nn.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    B = B.astype(jnp.float32)
    C = C.astype(jnp.float32)

    x = jnp.zeros((batch, dim, dstate), dtype=A.dtype)
    deltaA = jnp.exp(jnp.einsum("bdl,dn->lbdn", delta, A))
    deltaB_u = jnp.einsum("bdl,bnl,bdl->lbdn", delta, B, u)
    C = rearrange(C, "b n l -> l b n")

    last_state = None

    def f(x, inputs):
        deltaA, deltaB_u, C = inputs
        x = deltaA * x + deltaB_u
        y = jnp.einsum("bdn,bn->bd", x, C)
        return x, y

    last_state, y = jax.lax.scan(f, x, [deltaA, deltaB_u, C])
    y = rearrange(y, "l b d -> b d l")

    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * nn.activation.silu(z)
    out = out.astype(dtype_in)
    return out if not return_last_state else (out, last_state)
