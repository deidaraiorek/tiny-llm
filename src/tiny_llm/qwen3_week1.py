import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.rms_norm_eps = rms_norm_eps
        self.scale = head_dim**-0.5
        self.rope = RoPE(dims=head_dim, seq_len=max_seq_len,
                         base=theta, traditional=False)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        q = linear(x, self.wq)
        k = linear(x, self.wk)
        v = linear(x, self.wv)
        q = q.reshape(*q.shape[:-1], self.num_heads,
                      self.head_dim)      # -> (B, L, H_q, D)
        k = k.reshape(*k.shape[:-1], self.num_kv_heads,
                      self.head_dim)   # -> (B, L, H, D)
        v = v.reshape(*v.shape[:-1], self.num_kv_heads,
                      self.head_dim)   # -> (B, L, H, D)
        q = mx.fast.rms_norm(q, self.q_norm, self.rms_norm_eps)
        k = mx.fast.rms_norm(k, self.k_norm, self.rms_norm_eps)

        L = x.shape[-2]
        q = self.rope(q, offset=slice(0, L))
        k = self.rope(k, offset=slice(0, L))

        q = q.swapaxes(-3, -2)  # -> (B, H_q, L, D)
        k = k.swapaxes(-3, -2)  # -> (B, H, L, D)
        v = v.swapaxes(-3, -2)  # -> (B, H, L, D)

        orig_dtype = q.dtype
        q = q.astype(mx.float32)
        k = k.astype(mx.float32)
        v = v.astype(mx.float32)
        attn = scaled_dot_product_attention_grouped(
            q, k, v, scale=self.scale, mask=mask)
        attn = attn.astype(orig_dtype)

        attn = attn.swapaxes(-3, -2)  # -> (B, L, H_q, D)
        # -> (B, L, H_q*D)
        attn = attn.reshape(*attn.shape[:-2], self.num_heads * self.head_dim)
        return linear(attn, self.wo)


class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        up = linear(x, self.w_up)
        gate = silu(linear(x, self.w_gate))
        gated = gate * up
        return linear(gated, self.w_down)


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.attn = Qwen3MultiHeadAttention(
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            max_seq_len,
            theta,
            rms_norm_eps,
        )
        self.mlp = Qwen3MLP(hidden_size, intermediate_size,
                            w_gate, w_up, w_down)
        self.input_layernorm = RMSNorm(
            hidden_size, w_input_layernorm, rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            hidden_size, w_post_attention_layernorm, rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        h = x + self.attn(self.input_layernorm(x), mask)
        return h + self.mlp(self.post_attention_layernorm(h))


class Qwen3ModelWeek1:
    def __init__(self, mlx_model: Any):
        args = mlx_model.args
        self.hidden_size = args.hidden_size
        self.tie_word_embeddings = args.tie_word_embeddings

        inner = mlx_model.model
        self.embedding = Embedding(
            args.vocab_size,
            args.hidden_size,
            dequantize_linear(inner.embed_tokens).astype(mx.bfloat16),
        )

        self.layers = []
        for layer in inner.layers:
            attn = layer.self_attn
            mlp = layer.mlp
            self.layers.append(
                Qwen3TransformerBlock(
                    num_attention_heads=args.num_attention_heads,
                    num_kv_heads=args.num_key_value_heads,
                    hidden_size=args.hidden_size,
                    head_dim=args.head_dim,
                    intermediate_size=args.intermediate_size,
                    rms_norm_eps=args.rms_norm_eps,
                    wq=dequantize_linear(attn.q_proj).astype(mx.bfloat16),
                    wk=dequantize_linear(attn.k_proj).astype(mx.bfloat16),
                    wv=dequantize_linear(attn.v_proj).astype(mx.bfloat16),
                    wo=dequantize_linear(attn.o_proj).astype(mx.bfloat16),
                    q_norm=attn.q_norm.weight.astype(mx.bfloat16),
                    k_norm=attn.k_norm.weight.astype(mx.bfloat16),
                    w_gate=dequantize_linear(mlp.gate_proj).astype(mx.bfloat16),
                    w_up=dequantize_linear(mlp.up_proj).astype(mx.bfloat16),
                    w_down=dequantize_linear(mlp.down_proj).astype(mx.bfloat16),
                    w_input_layernorm=layer.input_layernorm.weight.astype(mx.bfloat16),
                    w_post_attention_layernorm=layer.post_attention_layernorm.weight.astype(mx.bfloat16),
                    max_seq_len=args.max_position_embeddings,
                    theta=args.rope_theta,
                )
            )

        self.norm = RMSNorm(
            args.hidden_size, inner.norm.weight.astype(mx.bfloat16), args.rms_norm_eps
        )

        if not args.tie_word_embeddings:
            self.lm_head = dequantize_linear(inner.lm_head).astype(mx.bfloat16)

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        h = self.embedding(inputs)
        for layer in self.layers:
            h = layer(h, mask="causal")
        h = self.norm(h)
        if self.tie_word_embeddings:
            return self.embedding.as_linear(h)
        return linear(h, self.lm_head)
