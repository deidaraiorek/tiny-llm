import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen3_week1 import Qwen3ModelWeek1
from .qwen3_week2 import Qwen3ModelWeek2
from typing import Callable


def simple_generate(
    model: Qwen3ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(model, y):
        logits = model(y)
        last_row = logits[:, -1, :]
        if sampler:
            next_token = sampler(last_row)
        else:
            next_token = mx.argmax(last_row, axis=-1)
        return next_token.item()

    tokens = tokenizer.encode(prompt)
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    for t in tokens:
        detokenizer.add_token(t)

    while True:
        y = mx.array(tokens).reshape(1, -1)
        next_token = _step(model, y)
        tokens.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break
        detokenizer.add_token(next_token)
        print(detokenizer.last_segment, end="", flush=True)

    detokenizer.finalize()
    print(detokenizer.last_segment, end="", flush=True)
    return detokenizer.text


def simple_generate_with_kv_cache(
    model: Qwen3ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        pass


def speculative_generate(
    draft_model: Qwen3ModelWeek2,
    model: Qwen3ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass
