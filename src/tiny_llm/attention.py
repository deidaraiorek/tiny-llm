import mlx.core as mx
from .basics import softmax, linear
import math 


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    d_k = query.shape[-1]
    cross_product = query @ key.swapaxes(-1, -2)
    if scale is not None:
        cross_product *= scale 
    else:
        cross_product /= math.sqrt(d_k)
    if mask is not None:
        cross_product += mask
    return softmax(cross_product,-1) @ value 

class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int, 
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dim = int(hidden_size / num_heads)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        q = linear(query, self.wq)
        k = linear(key, self.wk)
        v = linear(value, self.wv)
        q = q.reshape(*q.shape[:-1], self.num_heads, self.dim).swapaxes(-3, -2)
        k = k.reshape(*k.shape[:-1], self.num_heads, self.dim).swapaxes(-3,-2)
        v = v.reshape(*v.shape[:-1], self.num_heads, self.dim).swapaxes(-3, -2)
        attn = scaled_dot_product_attention_simple(q, k, v, None , mask)
        attn = attn.swapaxes(-3, -2).reshape(*query.shape[:-1], self.hidden_size)
        return linear(attn, self.wo)




def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
