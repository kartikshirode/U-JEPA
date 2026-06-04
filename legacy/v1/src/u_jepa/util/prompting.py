"""Chat-prompt formatting shared by the training and eval paths.

Qwen3-14B is a post-trained chat model AND a hybrid thinking model. Feeding it
raw text is off-distribution and, worse, on greedy decode it can emit a
<think>...</think> preamble before the answer, which makes a short eval window
miss the actual label and corrupts the accuracy / forgetting numbers.

format_chat_prompt wraps a user prompt with the tokenizer's chat template,
turns the generation prompt on, and disables thinking. Train and eval both call
it so the formatting is identical on both sides.

It degrades gracefully: if the tokenizer has no chat template (the tiny fake
tokenizers in the unit tests), it returns the raw prompt and reports that no
template was applied, so callers know whether to add special tokens themselves.
"""
from __future__ import annotations


def format_chat_prompt(tokenizer, prompt: str) -> tuple[str, bool]:
    """Return (formatted_prompt, used_template).

    used_template is True when the chat template was applied (so the special
    tokens are already in the string and the caller should tokenize with
    add_special_tokens=False), False when we fell back to the raw prompt
    (caller should add special tokens normally).
    """
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None:
        return prompt, False
    messages = [{"role": "user", "content": prompt}]
    # Newer Qwen tokenizers accept enable_thinking; older ones do not.
    for kwargs in (
        {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False},
        {"tokenize": False, "add_generation_prompt": True},
    ):
        try:
            text = apply(messages, **kwargs)
            if isinstance(text, str) and text:
                return text, True
        except TypeError:
            continue
        except Exception:
            break
    return prompt, False
