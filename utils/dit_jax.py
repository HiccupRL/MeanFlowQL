import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import variance_scaling  # 添加这一行
import numpy as np
import math
from typing import Any, Callable, Optional, Tuple, Sequence
from functools import partial


def kaiming_normal(scale=2.0):
    return variance_scaling(scale, 'fan_in', 'truncated_normal')

def kaiming_uniform(scale=2.0):
    return variance_scaling(scale, 'fan_in', 'uniform')

def default_init(scale=1.0):
    return nn.initializers.variance_scaling(scale, 'fan_in', 'truncated_normal')

class FeatureEmbed(nn.Module):
    """
    Feature Embedding layer for 1D vectors.
    """
    input_dim: int = 0
    embed_dim: int = 768
    norm_layer: Optional[Callable] = None

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.embed_dim, kernel_init=kaiming_normal(scale=0.01))(x)
        # Fix the problematic line
        if x.ndim == 1:
            x = x[None, None, :]
        elif x.ndim == 2:
            x = x[:, None, :]
        else:
            shape = x.shape[:-1] + (1, x.shape[-1])
            x = x.reshape(shape)
        if self.norm_layer is not None:
            x = self.norm_layer()(x)
        return x


def modulate(x, scale, shift):
    """Modulate the input with scale and shift."""
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    dim: int
    nfreq: int = 256
    
    @nn.compact
    def __call__(self, t):
        t_freq = self.timestep_embedding(t, self.nfreq)
        t_emb = nn.Sequential([
            nn.Dense(self.dim, kernel_init=kaiming_normal(scale=0.01)),
            nn.silu,
            nn.Dense(self.dim, kernel_init=kaiming_normal(scale=0.01))
        ])(t_freq)
        return t_emb
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        original_shape = t.shape
        if len(original_shape) > 2:
            t = t.reshape(-1, original_shape[-1])
        elif len(original_shape) == 1:
            t = t[:, None]
        
        half_dim = dim // 2
        freqs = jnp.exp(
            -math.log(max_period) * 
            jnp.arange(start=0, stop=half_dim, dtype=jnp.float32) / 
            half_dim
        )
        freqs = freqs[None, :]
        args = t * freqs
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    dim: int
    eps: float = 1e-6
    
    @nn.compact
    def __call__(self, x):
        scale = self.dim**0.5
        g = self.param('g', nn.initializers.ones, (1, 1, self.dim))
        
        norm_x = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        x_normed = x / norm_x
        
        return x_normed * scale * g


class MlpBlock(nn.Module):
    """Transformer MLP block."""
    dim: int
    mlp_dim: int
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x, deterministic=True):
        y = nn.Dense(self.mlp_dim, kernel_init=kaiming_normal(scale=0.01))(x)
        y = nn.gelu(y, approximate=True)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        y = nn.Dense(self.dim, kernel_init=kaiming_normal(scale=0.01))(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        return y


class Attention(nn.Module):
    """Multi-head attention."""
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    qk_norm: bool = False
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    norm_layer: Callable = RMSNorm
    
    @nn.compact
    def __call__(self, x, deterministic=True):
        batch_size, seq_len, width = x.shape
        head_dim = self.dim // self.num_heads
        scale = head_dim ** -0.5
        
        qkv = nn.Dense(
            features=self.dim * 3,
            use_bias=self.qkv_bias,
            kernel_init=kaiming_normal(scale=0.01),
            name='qkv'
        )(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.qk_norm:
            q = self.norm_layer(dim=head_dim)(q)
            k = self.norm_layer(dim=head_dim)(k)
        
        attn = (q @ jnp.transpose(k, (0, 1, 3, 2))) * scale
        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(rate=self.attn_drop)(attn, deterministic=deterministic)
        
        x = jnp.transpose(attn @ v, (0, 2, 1, 3))
        x = x.reshape(batch_size, seq_len, self.dim)
        x = nn.Dense(features=self.dim, kernel_init=kaiming_normal(scale=0.01))(x)
        x = nn.Dropout(rate=self.proj_drop)(x, deterministic=deterministic)
        
        return x


def zero_init():
    return nn.initializers.zeros

class DiTBlock(nn.Module):
    """DiT Transformer block with adaptive layer norm."""
    dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_norm: bool = True
    proj_drop: float = 0.0
    attn_drop: float = 0.0
    drop_path: float = 0.0
    act_layer: Callable = nn.gelu
    norm_layer: Callable = RMSNorm
    
    @nn.compact
    def __call__(self, x, c, deterministic=True):
        
        adaLN_modulation = nn.Sequential([
            nn.silu,
            nn.Dense(6 * self.dim, kernel_init=zero_init())  # Hiccup: Use small variance scaling init
        ])(c)
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            adaLN_modulation, 6, axis=-1
        )
        norm1 = self.norm_layer(self.dim)(x)
        norm1 = modulate(norm1, scale_msa, shift_msa)
        
        attn = Attention(
            dim=self.dim,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            qk_norm=self.qk_norm,
            attn_drop=self.attn_drop,
            proj_drop=self.proj_drop,
            norm_layer=self.norm_layer
        )(norm1, deterministic=deterministic)
        
        x = x + gate_msa[:, None, :] * attn
        
        norm2 = self.norm_layer(self.dim)(x)
        norm2 = modulate(norm2, scale_mlp, shift_mlp)
        
        mlp = MlpBlock(
            dim=self.dim,
            mlp_dim=int(self.dim * self.mlp_ratio),
            dropout_rate=self.proj_drop
        )(norm2, deterministic=deterministic)
        
        x = x + gate_mlp[:, None, :] * mlp
        
        return x


class FinalLayer(nn.Module):
    """Final layer of DiT."""
    dim: int
    out_dim: int
    init_scale: float = 1e-2  
    
    @nn.compact
    def __call__(self, x, c):
        adaLN_modulation = nn.Sequential([
            nn.silu,
            nn.Dense(2 * self.dim, kernel_init=zero_init())  
        ])(c)
        
        shift, scale = jnp.split(adaLN_modulation, 2, axis=-1)
        
        x = modulate(RMSNorm(self.dim)(x), scale, shift)
        
        x = nn.Dense(
            self.out_dim, 
            kernel_init=zero_init() 
        )(x)
        
        return x


class MFDiT(nn.Module):
    """
    Diffusion Transformer (DiT) model for 1D vector data.
    """
    input_dim: int = 0  # Set to 0 to infer from input
    hidden_dim: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    output_dim: int = 0  
    encoder: Optional[nn.Module] = None
    tanh_squash: bool = False  # Changed from True to False
    final_fc_init_scale: float = 0.0  # Changed from 1e-4 to 0.0
    use_output_layernorm: bool = False  # Changed from True to False
    gradient_clip_norm: float = 1.0
    residual_scale: float = 0.1
    
    def setup(self):
        # Dual timestep embedders (keeping this unique feature)
        self.t_embedder = TimestepEmbedder(dim=self.hidden_dim)
        self.r_embedder = TimestepEmbedder(dim=self.hidden_dim)
        
        # Positional embedding
        self.pos_embed = self.param(
            'pos_embed',
            nn.initializers.normal(stddev=0.005), 
            (1, 1, self.hidden_dim)
        )
        
        # Transformer blocks
        self.blocks = [
            DiTBlock(
                dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio
            )
            for _ in range(self.depth)
        ]
        
        # Input encoder (simplified like MFDiT_SIM)
        self.use_dynamic_embedder = (self.input_dim == 0)
        self.feature_embedder = None
        if self.use_dynamic_embedder:
            # Embedder will be created dynamically per input
            pass
        else:
            self.feature_embedder = FeatureEmbed(
                input_dim=self.input_dim,
                embed_dim=self.hidden_dim
            )

        # Final output dim fallback
        if self.output_dim == 0:
            self.output_dim = self.input_dim if self.input_dim > 0 else None

        # Final layer
        if self.output_dim is not None:
            self.final_layer = FinalLayer(
                dim=self.hidden_dim,
                out_dim=self.output_dim,
                init_scale=self.final_fc_init_scale
            )
        else:
            self.final_layer = None  # Delay create
    
    @nn.compact
    def __call__(self, observations, actions, r=None, t=None, is_encoded=False, train=True):
        """
        Forward pass of DiT.
        observations: Observation values
        actions: Action values
        r: Additional timestep (optional)
        t: Diffusion timestep (optional)
        is_encoded: Whether observations are already encoded
        train: Whether in training mode
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
            
        # Combine obs and action
        x_in = jnp.concatenate([observations, actions], axis=-1)
        
        # Dynamically create embedder and final layer if needed
        if self.use_dynamic_embedder:
            embedder = FeatureEmbed(input_dim=x_in.shape[-1], embed_dim=self.hidden_dim)
            final_layer = FinalLayer(
                dim=self.hidden_dim,
                out_dim=x_in.shape[-1] if self.output_dim is None else self.output_dim,
                init_scale=self.final_fc_init_scale
            )
        else:
            embedder = self.feature_embedder
            final_layer = self.final_layer

        # Embed
        x = embedder(x_in) + self.pos_embed  # shape: [B, 1, H]
        
        # Dual timestep embedding (keeping this unique feature)
        if t is not None and r is not None:
            t_emb = self.t_embedder(t)
            r_emb = self.r_embedder(r)
            c = t_emb + r_emb
        elif t is not None:
            # Fallback to single timestep if only t is provided
            t_emb = self.t_embedder(t)
            c = t_emb
        elif r is not None:
            # Fallback to single timestep if only r is provided
            r_emb = self.r_embedder(r)
            c = r_emb
        else:
            c = jnp.zeros((x.shape[0], self.hidden_dim))
        
        # Transformer blocks with unified residual mix (like MFDiT_SIM)
        for block in self.blocks:
            x_res = x
            x = block(x, c=c, deterministic=not train)
            x = x_res + self.residual_scale * (x - x_res)

        # Final projection
        x = final_layer(x, c=c)
        
        # Collapse sequence dimension
        x = x[:, 0, :]
        
        if self.use_output_layernorm:
            x = nn.LayerNorm(epsilon=1e-5)(x)
        
        return x



# new version .
class MFDiT_SIM(nn.Module):
    """
    Diffusion Transformer (DiT) model for 1D vector data (Flow Matching policy net).
    """
    input_dim: int = 0  
    hidden_dim: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    output_dim: int = 0  
    encoder: Optional[nn.Module] = None
    tanh_squash: bool = False  
    final_fc_init_scale: float = 0.0
    use_output_layernorm: bool = False
    gradient_clip_norm: float = 1.0
    residual_scale: float = 0.1

    def setup(self):
        # Timestep embedding
        self.t_embedder = TimestepEmbedder(dim=self.hidden_dim)

        # Positional embedding
        self.pos_embed = self.param(
            'pos_embed',
            nn.initializers.normal(stddev=0.005),
            (1, 1, self.hidden_dim)
        )

        # Transformer blocks
        self.blocks = [
            DiTBlock(
                dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio
            )
            for _ in range(self.depth)
        ]

        # Input encoder (optional)
        self.use_dynamic_embedder = (self.input_dim == 0)
        self.feature_embedder = None
        if self.use_dynamic_embedder:
            # Embedder will be created dynamically per input
            pass
        else:
            self.feature_embedder = FeatureEmbed(
                input_dim=self.input_dim,
                embed_dim=self.hidden_dim
            )

        # Final output dim fallback
        if self.output_dim == 0:
            self.output_dim = self.input_dim if self.input_dim > 0 else None

        # Final layer
        if self.output_dim is not None:
            self.final_layer = FinalLayer(
                dim=self.hidden_dim,
                out_dim=self.output_dim,
                init_scale=self.final_fc_init_scale
            )
        else:
            self.final_layer = None  # Delay create

    @nn.compact
    def __call__(self, observations, actions, t=None, is_encoded=False, train=True):
        """
        observations: [..., obs_dim]
        actions: [..., act_dim]
        t: timestep scalar tensor [..., 1] in [0, 1] (normalized)
        is_encoded: whether encoder(obs) has been applied
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)

        # Combine obs and action
        x_in = jnp.concatenate([observations, actions], axis=-1)

        # Dynamically create embedder and final layer if needed
        if self.use_dynamic_embedder:
            embedder = FeatureEmbed(input_dim=x_in.shape[-1], embed_dim=self.hidden_dim)
            final_layer = FinalLayer(
                dim=self.hidden_dim,
                out_dim=x_in.shape[-1] if self.output_dim is None else self.output_dim,
                init_scale=self.final_fc_init_scale
            )
        else:
            embedder = self.feature_embedder
            final_layer = self.final_layer

        # Embed
        x = embedder(x_in) + self.pos_embed  # shape: [B, 1, H]

        # Time embedding
        if t is not None:
            t_emb = self.t_embedder(t)
        else:
            t_emb = jnp.zeros((x.shape[0], self.hidden_dim))

        # Transformer blocks with residual mix
        for block in self.blocks:
            x_res = x
            x = block(x, c=t_emb, deterministic=not train)
            x = x_res + self.residual_scale * (x - x_res)

        # Final projection
        x = final_layer(x, c=t_emb)

        # Collapse sequence dimension
        x = x[:, 0, :]  # [B, D]

        if self.use_output_layernorm:
            x = nn.LayerNorm(epsilon=1e-5)(x)

        return x

class MFDiT_SIM_e(nn.Module):
    """
    Diffusion Transformer (DiT) model for 1D vector data (Flow Matching policy net).
    """
    input_dim: int = 0  
    hidden_dim: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    output_dim: int = 0  
    encoder: Optional[nn.Module] = None
    tanh_squash: bool = False  
    final_fc_init_scale: float = 0.0
    use_output_layernorm: bool = False
    gradient_clip_norm: float = 1.0
    residual_scale: float = 0.1

    def setup(self):
        # Timestep embedding
        self.t_embedder = TimestepEmbedder(dim=self.hidden_dim)

        # Positional embedding
        self.pos_embed = self.param(
            'pos_embed',
            nn.initializers.normal(stddev=0.005),
            (1, 1, self.hidden_dim)
        )

        # Transformer blocks
        self.blocks = [
            DiTBlock(
                dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio
            )
            for _ in range(self.depth)
        ]

        # Input encoder (optional)
        self.use_dynamic_embedder = (self.input_dim == 0)
        self.feature_embedder = None
        if self.use_dynamic_embedder:
            # Embedder will be created dynamically per input
            pass
        else:
            self.feature_embedder = FeatureEmbed(
                input_dim=self.input_dim,
                embed_dim=self.hidden_dim
            )

        # Final output dim fallback
        if self.output_dim == 0:
            self.output_dim = self.input_dim if self.input_dim > 0 else None

        # Final layer
        if self.output_dim is not None:
            self.final_layer = FinalLayer(
                dim=self.hidden_dim,
                out_dim=self.output_dim,
                init_scale=self.final_fc_init_scale
            )
        else:
            self.final_layer = None  # Delay create

    @nn.compact
    def __call__(self, observations, actions, t=None, is_encoded=False, train=True):
        """
        observations: [..., obs_dim]
        actions: [..., act_dim]
        t: timestep scalar tensor [..., 1] in [0, 1] (normalized)
        is_encoded: whether encoder(obs) has been applied
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)

        # Combine obs and action
        x_in = jnp.concatenate([observations, actions,t], axis=-1)

        # Dynamically create embedder and final layer if needed
        if self.use_dynamic_embedder:
            embedder = FeatureEmbed(input_dim=x_in.shape[-1], embed_dim=self.hidden_dim)
            final_layer = FinalLayer(
                dim=self.hidden_dim,
                out_dim=x_in.shape[-1] if self.output_dim is None else self.output_dim,
                init_scale=self.final_fc_init_scale
            )
        else:
            embedder = self.feature_embedder
            final_layer = self.final_layer

        # Embed
        x = embedder(x_in) + self.pos_embed  # shape: [B, 1, H]

        # Time embedding
        if t is not None:
            t_emb = self.t_embedder(t)
        else:
            t_emb = jnp.zeros((x.shape[0], self.hidden_dim))

        # Transformer blocks with residual mix
        for block in self.blocks:
            x_res = x
            x = block(x, c=t_emb, deterministic=not train)
            x = x_res + self.residual_scale * (x - x_res)

        # Final projection
        x = final_layer(x, c=t_emb)

        # Collapse sequence dimension
        x = x[:, 0, :]  # [B, D]

        if self.use_output_layernorm:
            x = nn.LayerNorm(epsilon=1e-5)(x)

        return x
