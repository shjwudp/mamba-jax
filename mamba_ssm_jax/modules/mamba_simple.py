import math
from typing import Callable
from collections import namedtuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from einops import rearrange, repeat


class Mamba(nn.Module):
    d_model: int
    d_inner: int
    d_state: int = 16
    d_conv: int = 4
    bias: bool = False
    c_dt_rank: str = "auto"
    conv_bias: bool = True
    use_fast_path: bool = False

    def setup(self):
        self.dt_rank = math.ceil(self.d_model / 16) if self.c_dt_rank == "auto" else self.c_dt_rank
        self.in_proj = nn.Dense(self.d_inner * 2, use_bias=self.bias)
        self.conv1d = nn.Conv(
            features=self.d_inner,
            use_bias=self.conv_bias,
            kernel_size=(self.d_conv,),
            feature_group_count=self.d_inner,
            padding="SAME",
        )
        self.x_proj = nn.Dense(self.dt_rank + self.d_state * 2, use_bias=False)
        self.dt_proj = nn.Dense(self.d_inner, use_bias=True)

        # S4D real initialization
        A = repeat(
            jnp.arange(1, self.d_state + 1, dtype=jnp.float32),
            "n -> d n",
            d=self.d_inner
        )
        self.A_log = jnp.log(A)

        # D "skip" parameter
        self.D = jnp.ones(self.d_inner)

        self.out_proj = nn.Dense(self.d_model, use_bias=self.bias, kernel_init=nn.initializers.kaiming_uniform())

    @nn.compact
    def __call__(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None

        xz = self.in_proj(hidden_states)

        A = -jnp.exp(self.A_log.astype(jnp.float32)) # (d_inner, d_state)
        assert not self.use_fast_path
        xz = rearrange(xz, "b l (d i) -> i b l d", i=2)
        x, z = xz[0], xz[1]
        x = self.conv1d(x)
        x = nn.activation.silu(x)

        x_dbl = self.x_proj(x)  # (b l d)
        dt, B, C = jnp.split(x_dbl, [self.dt_rank, self.dt_rank + self.d_state], axis=-1)
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
    dim: int
    mixer_cls: Callable
    mixer_kwargs: dict
    norm_cls: Callable
    fused_add_norm: bool = False
    residual_in_fp32: bool = False

    def setup(self):
        self.mixer = self.mixer_cls(**self.mixer_kwargs)
        self.norm = self.norm_cls(self.dim)

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
    dim: int
    n_layer: int
    vocab_size: int
    block_kwargs: dict
    rms_norm: bool = False

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.dim)
        self.layers = [Block(**self.block_kwargs) for i in range(self.n_layer)]
        self.norm_f = nn.LayerNorm(self.dim) if not self.rms_norm else nn.RMSNorm(self.dim)

    def __call__(self, input_ids):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.astype(hidden_states.dtype))
        return hidden_states


class MambaLMHeadModel(nn.Module):
    mixer_model_kwargs: dict

    def setup(self):
        self.backbone = MixerModel(**self.mixer_model_kwargs)

    def __call__(self, input_ids, position_ids=None):
        hidden_states = self.backbone(input_ids)
        lm_logits = self.backbone.embedding.attend(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)


def view_as_complex(arr):
    # Ensure the last dimension is of size 2
    assert arr.shape[-1] == 2, "Last dimension must be of size 2"

    # Split the last dimension into real and imaginary parts
    real, imag = jnp.split(arr, 2, axis=-1)

    # Create a complex array
    complex_array = real.squeeze(-1) + 1j * imag.squeeze(-1)

    return complex_array


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
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
    dtype_in = u.dtype
    u = u.astype(jnp.float32)
    delta = delta.astype(jnp.float32)

    if delta_softplus:
        delta = jax.nn.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    B = B.astype(jnp.float32)
    C = C.astype(jnp.float32)

    x = jnp.zeros((batch, dim, dstate), dtype=A.dtype)
    ys = []
    deltaA = jnp.exp(jnp.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = jnp.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    last_state = None
    L = u.shape[-1]
    for i in range(L):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        y = jnp.einsum("bdn,bn->bd", x, C[:, :, i])
        if i == L - 1:
            last_state = x
        ys.append(y)
    y = jnp.stack(ys, axis=2)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * nn.activation.silu(z)
    out = out.astype(dtype_in)
    return out if not return_last_state else (out, last_state)

